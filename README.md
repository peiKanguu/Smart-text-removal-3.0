# Smart-text-removal-3.0

一个基于OCR文字识别技术与OpenCV的用于批量识别并去除图片上文字的项目。

---

## 📌 流程

> 输入图片 → 文字检测 → 生成掩码 → 图像修复 → 输出图片

---

## 📂 项目结构（按模块划分）
```
SmartTextRemoval3.0/
├── datasets/                    # 输入图片存放位置
│   └── input_images/
├── outputs/                     # 输出目录
│   ├── detection_logs/          # 日志保存（检测结果、坐标、置信度等）
│   ├── mask_debug/              # 可视化的文字mask
│   └── cleaned_images/          # 最终清理后的图片
├── detect/                      # 模块1：OCR文字识别
│   └── detect_text.py           # 封装PaddleOCR或Tesseract调用
├── utils/                       # 模块2：掩码生成工具
│   └── mask_generator.py        # 根据OCR坐标生成掩码（OpenCV绘图）
├── lama_cleaner_model/         # 模块3：修复模型调用（如LaMa）
│   └── run_lama_cleaner.py      # 调用修复模型填补被删除文字
├── main.py                      # 主程序，串联整个流程
├── requirements.txt             # 所需依赖
└── README.md                    # 项目说明
```

---

## 📦 模块及工作流
| 状态 | 模块 | 说明 |
|:---:|:---:|:---:|
| [] | OCR识别 | detect_text.py 文本框识别与坐标提取
| [] | 掩码生成 | mask_generator.py OpenCV 绘制 mask
| [] | 图像修复 | 使用LaMa模型修复内容

---

## 🛠️ 待办建议（可选）

- [ ] 
- [ ] 
- [ ] 

---

## 📎 安装依赖

```bash
pip install -r requirements.txt

