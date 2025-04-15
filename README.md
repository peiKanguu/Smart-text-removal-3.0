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
├── lama_cleaner_model/          # 模块3：修复模型调用（如LaMa）
│   └── run_lama_cleaner.py      # 调用修复模型填补被删除文字
├── lama_env/                    # 虚拟环境目录（可忽略上传，仅本地生成）
├── main.py                      # 主程序，串联整个流程
├── setup_env.bat                # 一键创建并激活虚拟环境 + 安装依赖的脚本
├── run_all.bat                  # 一键运行主程序的脚本
├── requirements.txt             # 所需依赖
├── requirements-comment.txt             # 所需依赖的说明文档
└── README.md                    # 项目说明

```

---

## 📦 模块及工作流
| 状态 | 模块 | 说明 |
|:---:|:---:|:---:|
| [] | OCR识别 | detect_text.py 文本框识别与坐标提取
| [] | 掩码生成 | mask_generator.py OpenCV 绘制 mask
| [] | 图像修复 | 使用LaMa模型修复内容

#### 环境搭建
双击setup_env.bat(windows)
- 创建了名为 lama_env 的虚拟环境（隔离 Python 依赖）
- 自动激活该环境
- 自动执行：pip install -r requirements.txt
- 全部依赖安装成功后，提示用户安装完成

#### 运行
双击run_all.bat
- 自动激活虚拟环境
- 自动执行 main.py

#### 主程序逻辑
**第一阶段**
- 去路径 datasets/input_images/下批量读取图片
- 使用 PaddleOCR 识别文字
- 提取每张图片的检测结果（文字 + 坐标 + 置信度）保存到日志中



#### OCR文字识别
- 将一张图片输入，输出识别出来的文字 + 位置 + 置信度
- 分别输出到日志中以及控制台上
- 日志输出格式
```c
[
    {
        "text": "你好世界",
        "score": 0.9784,
        "box": [[100, 50], [200, 50], [200, 100], [100, 100]]
    },
    ...
]
```
- 控制台输出格式
```c
📄 正在处理：test1.png - 🔍 识别到 3 个文字区域
   ✏️ [1] "特价优惠"（置信度: 0.98）
   ✏️ [2] "立即抢购"（置信度: 0.95）
   ✏️ [3] "仅限今天"（置信度: 0.88）
```

---

## 🛠️ 待办建议（可选）

- [ ] 
- [ ] 
- [ ] 

---

## 📎 安装依赖

```bash
pip install -r requirements.txt

