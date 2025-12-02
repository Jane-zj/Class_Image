#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import uuid
import io
import json
import os
import requests
import torch
import torch.nn as nn
import urllib.parse

from flask import Flask
from flask_restx import Api, Resource, fields
from torchvision import transforms, models
from PIL import Image

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]  # 只输出到控制台
)

logger = logging.getLogger("classification_api")


# ====== 应用和文档初始化 ======
app = Flask(__name__)
api = Api(
    app,
    version="1.0",
    title="图像分类与存储 API",
    description="接收 uuid 和 图片 URL，下载并保存图片，返回 uuid/url/primary/secondary",
    doc="/docs"               # Swagger UI 地址 -> http://host:port/docs
)

# 把 Flask 的日志传递给 Gunicorn
gunicorn_logger = logging.getLogger('gunicorn.error')
logger.handlers = gunicorn_logger.handlers
logger.setLevel(gunicorn_logger.level)
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

# ====== 配置保存目录 ======
# SAVE_ROOT = '/home/ubuntu/classification/uuid'
# os.makedirs(SAVE_ROOT, exist_ok=True)

# ====== 加载模型与类别 ======
ANNOTATIONS_FILE = 'balanced_records.json'
WEIGHTS_PATH     = 'resnet50_best.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)
classes = sorted({r["pred_label"] for r in records})

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
state = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(state)
model.to(device).eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ====== 定义分类层级映射 ======
category_hierarchy = {
    "宝贝/亲子": ["出生纪念", "成长纪念", "童年时光", "孕妈主题", "夏令营"],
    "恋爱/写真": ["恋爱纪念", "个人写真", "闺蜜写真"],
    "婚纱/婚庆": [],
    "摄影/旅行": [],
    "全家福": [],
    "青春毕业季": ["宿舍情", "班级体"],
    "儿童毕业季": ["幼儿园", "通用"],
    "毕业/聚会": ["同学聚会", "中老年聚会"],
    "宠物写真": [],
    "退休/离职": [],
    "企业活动": [],
    "生日": ["儿童生日", "成人生日", "祝寿"],
}
secondary_to_primary = {
    sec: prim
    for prim, secs in category_hierarchy.items()
    for sec in secs
}

# ====== 定义请求/响应模型 ======
ns = api.namespace("classification", description="接收 uuid 和 URL，保存图片并分类返回")

payload = api.model("Payload", {
    "uuid": fields.String(required=True, description="请求的唯一标识 UUID"),
    "url":  fields.String(required=True, description="图片 URL 示例： https://…/img.jpg")
})
# result = api.model("Result", {
#     "uuid":      fields.String(description="请求的 UUID"),
#     "url":       fields.String(description="原始图片 URL"),
#     "primary":   fields.String(description="一级分类"),
#     "secondary": fields.String(description="二级分类（如果存在，否则为空）"),
#     "save_path": fields.String(description="图片保存的本地路径")
# })

result = api.model("Result", {
    "uuid":      fields.String(description="请求的 UUID"),
    "url":       fields.String(description="原始图片 URL"),
    "primary":   fields.String(description="一级分类"),
    "secondary": fields.String(description="二级分类（如果存在，否则为空）")
})

# ====== 路由实现 ======
@ns.route("/predict")
class Predict(Resource):
    @api.expect(payload, validate=True)
    @api.marshal_with(result)
    def post(self):
        """接收 JSON {uuid, url}，下载图片保存到 uuid 文件夹，返回 {uuid, url, primary, secondary, save_path}"""
        data = api.payload
        req_uuid = data["uuid"]
        img_url = data["url"]
        logger.info("收到请求: uuid=%s, url=%s", req_uuid, img_url)

        # target_dir = os.path.join(SAVE_ROOT, req_uuid)
        # os.makedirs(target_dir, exist_ok=True)
        try:
            resp = requests.get(img_url, timeout=5)
            resp.raise_for_status()
            # path = urllib.parse.urlparse(img_url).path
            # filename = os.path.basename(path) or f"{uuid.uuid4()}.jpg"
            # save_path = os.path.join(target_dir, filename)
            # with open(save_path, 'wb') as fw:
            #     fw.write(resp.content)
            #
            # img = Image.open(save_path).convert("RGB")
            # 从内存中直接读取图片进行处理
            img_bytes = io.BytesIO(resp.content)
            img = Image.open(img_bytes).convert("RGB")
            inp = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp)
                idx = out.argmax(1).item()

            pred = classes[idx]
            if pred in secondary_to_primary:
                primary   = secondary_to_primary[pred]
                secondary = pred
            else:
                primary   = pred
                secondary = ""
            logger.info("预测完成: uuid=%s, primary=%s, secondary=%s",
                        req_uuid, primary, secondary)

            return {
                "uuid":      req_uuid,
                "url":       img_url,
                "primary":   primary,
                "secondary": secondary,
                # "save_path": save_path
            }

        except Exception as e:
            logger.error("请求失败: uuid=%s, url=%s, 错误=%s", req_uuid, img_url, str(e))
            api.abort(500, f"服务器内部错误：{e}")

# ====== 启动 ======
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)
