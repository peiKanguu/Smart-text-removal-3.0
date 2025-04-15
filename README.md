# Smart-text-removal-3.0
一个基于OCR文字识别技术与OpenCV的用于批量识别并去除图片上文字的项目

# 流程
> 输入图片 → 文字检测 → 生成掩码 → 图像修复 → 输出图片

# 架构（按模块划分）
📦 SmartTextRemoval3.0/
│
├── 📁 datasets/                     # 输入图片存放位置
│   └── input_images/
│
├── 📁 outputs/                      # 输出目录
│   ├── detection_logs/             # 日志保存（检测结果、坐标、置信度等）
│   ├── mask_debug/                 # 可视化的文字mask
│   └── cleaned_images/             # 最终清理后的图片
│
├── 📁 detect/                       # 模块1：OCR文字识别
│   └── detect_text.py              # 封装PaddleOCR或Tesseract调用
│
├── 📁 utils/                        # 模块2：掩码生成工具
│   └── mask_generator.py           # 根据OCR坐标生成掩码（OpenCV绘图）
│
├── 📁 lama_cleaner_model/          # 模块3：修复模型调用（如LaMa）
│   └── run_lama_cleaner.py         # 调用修复模型填补被删除文字
│
├── main.py                         # 主程序，串联整个流程
├── requirements.txt                # 所需依赖
└── README.md                       # 项目说明

模块 | 名称 | 关键技术 | 功能
模块1 | detect_text.py | PaddleOCR / Tesseract | 对图像文字进行定位与识别，返回坐标框
模块2 | mask_generator.py | OpenCV | 根据OCR返回的框生成二值掩码图
模块3 | run_lama_cleaner.py | LaMa (PyTorch模型) | 使用掩码对目标区域进行内容感知修复
主控 | main.py | tqdm, os, logging | 批量处理图片、管理IO路径、保存日志
