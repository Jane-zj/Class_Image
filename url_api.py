#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能照片书排序工具 v10.8 —— Flask API 版（纯内存处理）
- 基于 v10.7 版本优化，移除了所有文件保存功能，仅通过 API 返回结果。
- 主要改动：
  1) 移除 `save_visual_results` 函数，不再将图片保存到本地。
  2) 移除将 AI 分析结果保存为 JSON 文件的功能。
  3) 更新了 API 响应消息，以反映变化。
  4) 所有核心排序逻辑保持不变。

运行：
  pip install flask flask-restx httpx Pillow tqdm json-repair openai
  python your_script_name.py  # 默认监听 0.0.0.0:5002

注意：
  - 本服务会调用自定义 OpenAI 兼容接口（见 API_CONFIG_*），请替换为你自己的地址与 Key。
  - 若部署在生产环境，建议使用 gunicorn/uvicorn + gevent/uvloop 等方式托管。
"""

import sys
import base64
import json
import io
import time
import uuid
import asyncio
import logging
from typing import List, Dict, Any

import httpx
from json_repair import repair_json as repair

from flask import Flask
from flask_restx import Api, Resource, fields

# ================== 第三方库可选导入 =====================
try:
    from tqdm import tqdm as sync_tqdm
    from openai import AsyncOpenAI
    from PIL import Image, UnidentifiedImageError
except ImportError as e:
    print("错误: 关键库缺失，请先运行 'pip install openai httpx Pillow tqdm json-repair flask flask-restx'。详情:", e)
    sys.exit(1)

# ================== 全局配置 =====================
# 全局日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [Sorter-API] - %(message)s',
                    handlers=[logging.FileHandler("photo_sorter_api.log", mode='a'), logging.StreamHandler()])

# API配置 (请替换为您自己的Key)
API_CONFIG_ANALYSIS = {
    "api_key": "YOUR_ANALYSIS_API_KEY",
    "base_url": "http://117.50.172.250:6006/v1"
}
API_CONFIG_TIMELINE = {
    "api_key": "YOUR_TIMELINE_API_KEY",
    "base_url": "http://117.50.172.250:6006/v1"
}

# 全局调优参数
TUNING_PARAMETERS = {
    "concurrency_limit": 10,  # 并发Worker的数量
    "multi_image_batch_size": 5,
    "resize_to": 512,
    "jpeg_quality": 75,
    "request_timeout": httpx.Timeout(60.0, read=360.0, write=60.0),
    "download_timeout": httpx.Timeout(30.0),
    "max_retries": 2  # 为每张图片设定的最大重试次数
}

# ================== 提示词 (Prompts) =====================
PROMPT_MULTI_IMAGE = """
你是一位极其严谨且高效的AI场景分析师，擅长批处理图像。你将收到一个包含 N 张图片的序列。你的任务是为 **每一张** 图片提取一组用于“事件聚类”的【核心关键词】。为了让相似场景的照片能被精确地分到一组，你的关键词必须是客观、稳定且可复用的。

**关键词提取原则 (对每一张图片都适用):**

1.  **核心元素与活动**: 专注于图片最核心的地点和活动（如 '会议', '用餐', '滑雪'）。

2.  **地点标准化原则**: **必须为相似场景选择一个统一、标准的地点标签。** 例如，所有在雪地里的户外活动，统一使用 **'雪景户外'** 作为地标。

3.  **优先识别固定团体**: 如果照片中反复出现的是同一群人，**必须优先使用一个统一的团体关键词**，例如 `'家庭合影'`, `'团队活动'`。

4.  **稳定特征与光线**: 可以加入在整个事件中都保持不变的显著特征。光线只用 '白天', '夜晚', '黄昏', '室内光'。

5.  **绝对禁止主观与易变词汇**: 严禁使用任何主观感受、情绪、氛围或瞬间动作。例如：温馨, 快乐, 美丽, 特写, 抓拍。

**【【【警告：输出格式是唯一准则】】】**
你的回答 **必须是、且只能是** 一个RFC 8259标准下**完美无瑕的JSON对象**。
- **严禁** 在JSON的`{}`括号外添加任何说明、注释、标题或任何多余的文字。
- **严禁** 生成任何损坏的JSON（例如，缺少逗号、多了逗号、引号错误）。
- 你的整个输出必须能被 `json.loads()` 函数直接解析，否则任务即视为失败。

---
## 示例 ##
**你应该输出:**
```json
{
  "results": [
    {
      "image_id": "a1b2c3d4-e5f6-7890-1234-567891abcdef",
      "analysis": {
        "location_landmark": "餐厅",
        "main_keywords": ["男孩", "儿童", "用餐", "室内光"]
      }
    },
    {
      "image_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
      "analysis": {
        "location_landmark": "长城",
        "main_keywords": ["家庭合影", "长城", "白天"]
      }
    }
  ]
}
```
现在，请严格遵守以上所有原则，特别是关于输出格式的警告，分析接下来提供的图片。
"""

PROMPT_TIMELINE_SORTER = """
你是一个高度专业化的排序API。你的唯一功能是接收一个包含多个事件ID和描述的JSON对象，然后根据逻辑、常识和时间线索对这些事件ID进行重新排序。

【【【警告：输出格式是唯一准则】】】
返回纯净JSON数组: 你的回答 必须是、且只能是 一个标准的JSON数组（一个由[开始、]结束的字符串列表）。

严禁任何额外字符: 绝对禁止在JSON数组的[]括号前后添加任何解释、注释、代码块标记( ```json)或任何其他多余的文本。你的整个输出必须能被 json.loads() 函数直接解析。

禁止遗漏或新增: 返回的ID列表长度必须与输入的ID列表长度 完全相等。禁止新增、重复或遗漏任何ID。

自我验证步骤
在你最终确定输出之前，请在内部执行一次自我检查：
"我收到了 X 个事件ID，我即将返回的是一个纯净的、不带任何额外字符的、包含不多不少正好 X 个ID的JSON数组吗？"
如果答案是否定，你必须重新生成回答，直到完全符合要求为止。

示例
如果输入中包含3个事件ID，你的输出 必须是且仅是 如下所示的字符串: ["Event_3", "Event_1", "Event_2"]
现在，这里是需要你排序的 {event_count} 个事件：
{event_list_json}

请严格遵守以上所有规则，你的回答必须是且仅是一个可以直接被JSON解析的数组:
"""

# ================== 核心工具函数 =====================
async def download_image(url: str, semaphore: asyncio.Semaphore, client: httpx.AsyncClient):
    """使用信号量并发异步下载图片。"""
    async with semaphore:
        try:
            response = await client.get(url, timeout=TUNING_PARAMETERS["download_timeout"])
            response.raise_for_status()
            return url, response.content
        except httpx.RequestError as e:
            logging.warning(f"下载图片失败 {url}: {e}")
            return url, None


def process_and_encode_image(image_bytes: bytes, max_size: int, quality: int):
    """【智能缩放】处理图片字节流 (缩放、转换、编码为base64)。"""
    if not image_bytes:
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.height > max_size or img.width > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=quality, optimize=True)
            return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except (UnidentifiedImageError, IOError) as e:
        logging.warning(f"无法处理图片数据: {e}")
        return None


async def get_default_model_name(client: AsyncOpenAI, preferred_model_prefix="qwen-vl"):
    """异步从API获取默认的模型名称。"""
    try:
        models = await client.models.list()
        model_list = models.data if hasattr(models, "data") else models
        qwen_models = [m.id for m in model_list if hasattr(m, "id") and preferred_model_prefix in m.id]
        if qwen_models:
            return sorted(qwen_models, reverse=True)[0]
        if model_list and len(model_list) > 0 and hasattr(model_list[0], "id"):
            return model_list[0].id
    except Exception as e:
        logging.error(f"自动获取模型名称失败: {e}")
    return "qwen-vl-max"


def calculate_jaccard_similarity(set1, set2):
    """计算两个集合的Jaccard相似度。"""
    if not set1 and not set2:
        return 1.0
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    return intersection_size / union_size if union_size != 0 else 0.0


async def process_analysis_batch(batch_items, analysis_model_name, client_analysis):
    """处理一批图片进行AI分析（移除了semaphore，因为并发由Worker数量控制）。"""
    try:
        messages_content, valid_items_for_api = [], []
        for item in batch_items:
            base64_image = await asyncio.to_thread(process_and_encode_image, item['image_bytes'],
                                                   TUNING_PARAMETERS["resize_to"],
                                                   TUNING_PARAMETERS["jpeg_quality"])
            if base64_image:
                messages_content.append({'type': 'image_url', 'image_url': {'url': base64_image}})
                valid_items_for_api.append(item)

        if not valid_items_for_api:
            return [{"uuid": item['uuid'], "status": "error", "data": "编码失败"} for item in batch_items]

        id_list_text = "此批次的图片ID列表:\n" + "\n".join([item['uuid'] for item in valid_items_for_api])
        messages_content.append({"type": "text", "text": id_list_text})

        response = await client_analysis.chat.completions.create(
            model=analysis_model_name,
            messages=[
                {'role': 'system', 'content': PROMPT_MULTI_IMAGE},
                {'role': 'user', 'content': messages_content}
            ],
            temperature=0, max_tokens=10240
        )
        response_data = json.loads(repair(response.choices[0].message.content))
        if 'results' not in response_data or not isinstance(response_data['results'], list):
            raise ValueError("从AI接收到无效的JSON格式")

        ai_results_map = {item.get('image_id'): item.get('analysis') for item in response_data['results']}
        final_results = []
        for item in batch_items:
            if item['uuid'] in ai_results_map and ai_results_map[item['uuid']]:
                final_results.append({"uuid": item['uuid'], "status": "success", "data": ai_results_map[item['uuid']]})
            else:
                final_results.append({"uuid": item['uuid'], "status": "error", "data": "AI分析成功但此图片无返回结果"})
        return final_results
    except Exception as batch_error:
        logging.error(f"批处理严重失败 ({batch_error})。此批次所有图片都将标记为失败。")
        return [{"uuid": item['uuid'], "status": "error", "data": str(batch_error)} for item in batch_items]


async def generate_final_order(all_data, original_url_order, timeline_model_name, client_timeline):
    """根据AI分析结果对照片进行聚类和排序，生成最终顺序。"""
    summary = {"ai_sort_success": False, "final_event_count": 0, "final_order_map": None, "final_photo_list": []}
    if not all_data:
        return summary

    valid_photos = [item for item in all_data if item.get('status') == 'success']
    failed_photos = [item for item in all_data if item.get('status') != 'success']

    logging.info("根据AI分析结果进行图片聚类...")
    SIMILARITY_THRESHOLD = 0.4
    event_clusters_list = []
    for item in valid_photos:
        data = item.get('data', {})
        current_landmark = data.get('location_landmark', 'unknown').lower()
        current_keywords = set(k.lower() for k in data.get('main_keywords', []))

        best_match_cluster = None
        highest_similarity = -1
        for cluster in event_clusters_list:
            if cluster['landmark'] == current_landmark:
                similarity = calculate_jaccard_similarity(current_keywords, cluster['keywords_set'])
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match_cluster = cluster

        if best_match_cluster and highest_similarity >= SIMILARITY_THRESHOLD:
            best_match_cluster['photos'].append(item)
        else:
            event_clusters_list.append({'photos': [item], 'landmark': current_landmark, 'keywords_set': current_keywords})

    event_clusters = {f"Event_{i+1}": cluster_data for i, cluster_data in enumerate(event_clusters_list)}
    logging.info(f"聚类完成。共发现 {len(event_clusters)} 个不同事件。")
    summary["final_event_count"] = len(event_clusters)

    if not event_clusters:
        final_photo_list = failed_photos
    else:
        events_for_sorting = {}
        for event_id, cluster in event_clusters.items():
            items = cluster['photos']
            for item in items:
                if 'url' in item and item['url'] in original_url_order:
                    item['original_order'] = original_url_order.index(item['url'])
            items.sort(key=lambda x: x.get('original_order', -1))

            description = f"在[{cluster['landmark']}] " + " ".join(list(cluster['keywords_set']))
            events_for_sorting[event_id] = {
                "description": description.strip(), "photos": items,
                "avg_original_order": sum(p['original_order'] for p in items if 'original_order' in p) / len(items) if items else 0
            }

        logging.info("请求AI进行全局故事线编排...")
        event_list_for_prompt = {eid: data['description'] for eid, data in events_for_sorting.items()}
        sorted_event_ids = []

        try:
            timeline_prompt = PROMPT_TIMELINE_SORTER.format(
                event_count=len(event_list_for_prompt),
                event_list_json=json.dumps(event_list_for_prompt, ensure_ascii=False, indent=2)
            )
            response = await client_timeline.chat.completions.create(
                model=timeline_model_name, messages=[{'role': 'user', 'content': timeline_prompt}],
                temperature=0.0, max_tokens=10240
            )
            response_text = response.choices[0].message.content
            parsed_json = json.loads(repair(response_text))

            if isinstance(parsed_json, list) and all(isinstance(e, str) for e in parsed_json):
                sorted_event_ids = parsed_json
                summary["ai_sort_success"] = True
            else:
                raise ValueError("解析后的JSON不是一个字符串列表。")
        except Exception as e:
            logging.warning(f"AI故事线排序失败: {e}。回退到按原始顺序排序。")
            summary["ai_sort_success"] = False
            sorted_events = sorted(events_for_sorting.items(), key=lambda item: item[1]['avg_original_order'])
            sorted_event_ids = [item[0] for item in sorted_events]

        final_photo_list = []
        all_event_ids = set(events_for_sorting.keys())
        seen_ids = set()
        for event_id in sorted_event_ids:
            if event_id in events_for_sorting and event_id not in seen_ids:
                final_photo_list.extend(events_for_sorting[event_id]['photos'])
                seen_ids.add(event_id)

        missed_ids = sorted(list(all_event_ids - seen_ids), key=lambda eid: events_for_sorting[eid]['avg_original_order'])
        for event_id in missed_ids:
            final_photo_list.extend(events_for_sorting[event_id]['photos'])

        if failed_photos:
            failed_photos.sort(key=lambda x: original_url_order.index(x['url']) if 'url' in x and x['url'] in original_url_order else -1)
            final_photo_list.extend(failed_photos)
    summary['final_photo_list'] = final_photo_list
    logging.info(f"======== 最终列表检查 (总数: {len(final_photo_list)}) ========")
    for idx, item in enumerate(final_photo_list, 1):
        if 'url' not in item:
            logging.error(f"【【严重错误】】: 序号 {idx} 的元素缺少 'url' 键! 内容: {item}")
    # summary['final_order_map'] = {item['url']: idx for idx, item in enumerate(final_photo_list, 1) if 'url' in item}
    final_map = {} 
    for idx, item in enumerate(final_photo_list, 1):
        if 'url' in item:
            url = item['url']
            if url not in final_map:
                # 这是该 URL 第一次出现，正常赋值
                final_map[url] = idx
            else:
                # 这是重复的 URL
                if isinstance(final_map[url], list):
                    # 如果已经是一个列表，直接追加
                    final_map[url].append(idx)
                else:
                    # 如果是单个数字，将其转换为列表
                    final_map[url] = [final_map[url], idx]
    summary['final_order_map'] = final_map
    return summary

# ================== 并发 Worker ==================
async def analysis_worker(worker_id, queue, client_analysis, model_name, final_results, pbar):
    """从队列中获取批次，处理并根据结果决定重试或完成。"""
    while True:
        try:
            batch_items = await queue.get()

            batch_results = await process_analysis_batch(batch_items, model_name, client_analysis)

            uuid_to_item_map = {item['uuid']: item for item in batch_items}
            items_to_retry = []

            for result in batch_results:
                item = uuid_to_item_map[result['uuid']]
                if result['status'] == 'success':
                    # 注意：此处不再需要保留 image_bytes，因为不再保存文件，可以节省内存
                    # 但为了保持逻辑一致性，暂时保留，最终返回的 map 不会包含它
                    final_results[item['url']] = {"url": item['url'], **result}
                    pbar.update(1)
                else:
                    item['retry_count'] += 1
                    if item['retry_count'] < TUNING_PARAMETERS['max_retries']:
                        logging.warning(f"图片分析失败，将重试 (第 {item['retry_count']} 次): {item['url']}")
                        items_to_retry.append(item)
                    else:
                        logging.error(f"图片分析失败，已达最大重试次数: {item['url']}")
                        final_results[item['url']] = {"url": item['url'], **result}
                        pbar.update(1)

            # 需要重试的项重新打包成批次放回队列
            if items_to_retry:
                await queue.put(items_to_retry)

            queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.error(f"Worker {worker_id} 发生未知错误: {e}")
            queue.task_done()  # 避免死锁


async def run_sorting_pipeline(image_urls: List[str], request_id: str):
    """排序过程的主执行流程 (v10.8 纯内存版)。"""
    timings: Dict[str, float] = {}
    overall_start_time = time.time()

    # 复用一个 AI httpx 客户端，另一个用于下载
    async with httpx.AsyncClient() as http_client_general, \
            httpx.AsyncClient(limits=httpx.Limits(max_connections=TUNING_PARAMETERS["concurrency_limit"] + 5), timeout=TUNING_PARAMETERS["request_timeout"]) as http_client_ai:

        client_analysis = AsyncOpenAI(api_key=API_CONFIG_ANALYSIS["api_key"], base_url=API_CONFIG_ANALYSIS["base_url"], http_client=http_client_ai)
        client_timeline = AsyncOpenAI(api_key=API_CONFIG_TIMELINE["api_key"], base_url=API_CONFIG_TIMELINE["base_url"], http_client=http_client_ai)

        analysis_model_name = await get_default_model_name(client_analysis, "qwen-vl")
        timeline_model_name = await get_default_model_name(client_timeline, "qwen-plus")
        logging.info(f"分析模型: {analysis_model_name}, 排序模型: {timeline_model_name}")

        # --- 阶段 1: 图片下载 ---
        download_start = time.time()
        logging.info(f"[{request_id}] === 阶段 1: 开始下载 {len(image_urls)} 张图片 ===")
        semaphore = asyncio.Semaphore(TUNING_PARAMETERS['concurrency_limit'])
        download_tasks = [download_image(url, semaphore, http_client_general) for url in image_urls]

        downloaded_results_list = []
        pbar_download = sync_tqdm(total=len(download_tasks), desc="下载图片")
        for task in asyncio.as_completed(download_tasks):
            result = await task
            downloaded_results_list.append(result)
            pbar_download.update(1)
        pbar_download.close()
        downloaded_images_map = dict(downloaded_results_list)
        timings["download_phase"] = time.time() - download_start

        # --- 阶段 2: AI分析 (并发Worker模型) ---
        analysis_start = time.time()
        logging.info(f"[{request_id}] === 阶段 2: 开始AI分析 (并发Worker模型) ===")

        analysis_queue: asyncio.Queue = asyncio.Queue()
        final_analysis_results: Dict[str, Any] = {}

        items_to_process = []
        for url in image_urls:
            image_bytes = downloaded_images_map.get(url)
            if image_bytes:
                items_to_process.append({
                    'uuid': str(uuid.uuid4()), 'url': url,
                    'image_bytes': image_bytes, 'retry_count': 0
                })
            else:
                final_analysis_results[url] = {'url': url, 'status': 'error', 'data': '下载失败'}

        # [生产者] 将任务分批放入队列
        batch_size = TUNING_PARAMETERS['multi_image_batch_size']
        for i in range(0, len(items_to_process), batch_size):
            batch = items_to_process[i:i + batch_size]
            analysis_queue.put_nowait(batch)

        pbar_analysis = sync_tqdm(total=len(items_to_process), desc="AI分析任务")

        # [消费者] 创建并启动Workers
        workers = []
        for i in range(TUNING_PARAMETERS['concurrency_limit']):
            worker = asyncio.create_task(analysis_worker(
                f"Worker-{i + 1}", analysis_queue, client_analysis,
                analysis_model_name, final_analysis_results, pbar_analysis
            ))
            workers.append(worker)

        # 等待所有队列中的任务被处理完成
        await analysis_queue.join()

        # 所有任务完成后，取消所有worker
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        pbar_analysis.close()

        all_analysis_results = [final_analysis_results[url] for url in image_urls if url in final_analysis_results]
        timings["analysis_phase"] = time.time() - analysis_start
        
        # --- 阶段 3: 聚类与故事线排序 ---
        sort_start = time.time()
        logging.info(f"[{request_id}] === 阶段 3: 开始聚类与故事线排序 ===")
        # 注意：此处 all_analysis_results 不包含 image_bytes，但 generate_final_order 也不需要它
        sort_summary = await generate_final_order(all_analysis_results, image_urls, timeline_model_name, client_timeline)
        timings["final_order_phase"] = time.time() - sort_start
        
        # 阶段 4: 文件保存阶段已被移除
        logging.info(f"[{request_id}] === 流程结束 ===")

        timings["total"] = time.time() - overall_start_time
        return sort_summary.get("final_order_map"), timings


# ================== Flask + RESTX 定义 ==================
app = Flask(__name__)
api = Api(
    app,
    version="1.0",
    title="智能照片书排序 API (纯内存版)",
    description="将 URL 列表中的图片进行下载、AI分析、事件聚类与时间线排序，并输出最终顺序。",
    doc="/docs",  # Swagger UI 入口
)
ns = api.namespace("api/v1", description="排序相关接口")

# —— 数据模型（用于文档/参数校验） ——
url_list_model = api.model(
    "UrlListInput",
    {
        "image_urls": fields.List(fields.String, required=True, description="图片 URL 列表", example=[
            "[https://example.com/a.jpg](https://example.com/a.jpg)", "[https://example.com/b.png](https://example.com/b.png)"
        ])
    }
)

sort_result_model = api.model(
    "SortResult",
    {
        "request_id": fields.String(description="任务ID"),
        "final_order_map": fields.Raw(description="图片URL到序号的映射(从1开始)"),
        "timings": fields.Raw(description="各阶段耗时统计(秒)"),
        "message": fields.String(description="说明")
    }
)


@ns.route('/health')
class Health(Resource):
    @api.response(200, 'OK')
    def get(self):
        """健康检查"""
        return {"status": "ok"}, 200


@ns.route('/sort')
class SortPhotos(Resource):
    @api.expect(url_list_model, validate=True)
    @api.marshal_with(sort_result_model, code=200, as_list=False)
    def post(self):
        """提交一个图片URL列表，返回排序结果与耗时。"""
        payload = api.payload or {}
        image_urls: List[str] = payload.get('image_urls') or []
        
        if not image_urls or not isinstance(image_urls, list):
            api.abort(400, "请求体必须是一个包含 'image_urls' 键的JSON对象，且其值不能为空列表。")

        request_id = str(uuid.uuid4())
        
        # 在同步 Flask 处理函数中运行异步流水线
        try:
            final_order_map, timings = asyncio.run(run_sorting_pipeline(image_urls, request_id))
        except RuntimeError:
            # 若当前已有事件循环（例如在某些WSGI容器中），使用新循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            final_order_map, timings = loop.run_until_complete(run_sorting_pipeline(image_urls, request_id))
            loop.close()

        resp = {
            "request_id": request_id,
            "final_order_map": final_order_map or {},
            "timings": timings or {},
            "message": "排序处理完成。"
        }
        return resp, 200


if __name__ == '__main__':
    # 监听 0.0.0.0:5002
    app.run(host='0.0.0.0', port=5002, debug=True)
