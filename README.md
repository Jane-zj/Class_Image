# Class_Image
## 接收用户提交的 uuid 与图片 url，后台下载图片，使用图像分类模型对图片进行分类，并返回包含 uuid、原始 URL、分类的一级标题和二级标题的 JSON 响应。

├── label_api.py
├── balanced_records.json # 类别标注文件
└── resnet50_best.pth # 训练好的模型权重
