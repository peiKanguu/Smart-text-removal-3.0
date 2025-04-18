import os
import cv2
import json
from tqdm import tqdm
import numpy as np
np.int = int  # ✅ 解决 numpy.int 报错
from utils.mask_generator import generate_mask

# ✅ 安全导入 detect_text
try:
    from detect.detect_text import detect_text # OCR 模块
except Exception as e:
    print("❌ 导入模块失败:", e)
    exit(1)

# 配置路径
input_folder = './datasets/input_images'
output_log_folder = './outputs/detection_logs'
output_mask_folder = './outputs/mask_debug'

# 根据图片特征（尺寸 + 掩码特征）判断使用openCV的算法
def choose_inpaint_method(img, mask):
    h, w = img.shape[:2]
    img_area = h * w

    # 掩码白色像素个数（即需修复的区域）
    mask_area = np.sum(mask > 0)

    ratio = mask_area / img_area

    # 👇 可调节阈值，设置自动切换策略
    if ratio < 0.002 or max(h, w) < 400:
        # 小字体 + 小图像时，快速填色就够了
        print("🧠 策略判断：使用 INPAINT_TELEA（快速修复）")
        return cv2.INPAINT_TELEA
    else:
        print("🧠 策略判断：使用 INPAINT_NS（结构修复）")
        return cv2.INPAINT_NS

# 输出文件夹
os.makedirs(output_log_folder, exist_ok=True)

def process_image(img_path):
    import json
    img_name = os.path.basename(img_path)
    base_name = os.path.splitext(img_name)[0]

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 无法读取图像：{img_path}")
        return

    # 🧠 步骤 1：OCR识别
    detections = detect_text(img)  # list of dicts: {'text', 'score', 'bbox'}
    
    # ✅ 打印识别结果概览
    print(f"📄 正在处理：{img_name} - 🔍 识别到 {len(detections)} 个文字区域")
    
    # ✅ 打印每一条文字识别内容和置信度
    if detections:
        for i, item in enumerate(detections, 1):
            text = item['text']
            score = item['score']
            print(f"   ✏️ [{i}] \"{text}\"（置信度: {score:.2f}）")
    else:
        print("   ⚠️ 未检测到任何文字")
    
    # 📝 步骤 2：写入日志 JSON 文件
    log_data = {
        "filename": img_name,
        "detections": []
    }

    for det in detections:
        log_data["detections"].append({
            "text": det.get("text", ""),
            "score": round(det.get("score", 0.0), 4),
            "bbox": det.get("box", [])  # 用 .get 安全访问
        })

    log_path = os.path.join(output_log_folder, base_name + '_ocr_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    print(f"✅ {img_name} -> 识别结果已保存至日志")
    
    # 🧪 步骤 3：生成并保存调试用掩码图
    mask_path = os.path.join(output_mask_folder, base_name + '_mask.png')
    mask = generate_mask(img, detections, save_path=mask_path)
    
    # 🧽 步骤 4：使用 OpenCV 进行图像修复
    output_cleaned_folder = './outputs/cleaned_images'
    os.makedirs(output_cleaned_folder, exist_ok=True)
    output_cleaned_path = os.path.join(output_cleaned_folder, base_name + '_cleaned.png')

    # 读取掩码（灰度图）
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"❌ 无法读取掩码图像：{mask_path}")
        return

    # 自动选择修复策略
    method = choose_inpaint_method(img, mask)
    
    # 执行修复
    inpainted = cv2.inpaint(img, mask, inpaintRadius=3, flags=method)

    # 保存修复后图像
    cv2.imwrite(output_cleaned_path, inpainted)
    print(f"🖼️ 图像修复完成：{output_cleaned_path}")


if __name__ == "__main__":
    print("🚀 开始批量处理图片...")
    # ✅ 获取所有待处理图片路径
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"📸 待处理图片数量: {len(image_files)}")

    # ✅ 使用 tqdm 显示处理进度
    for filename in tqdm(image_files, desc="处理中"):
        img_path = os.path.join(input_folder, filename)
        process_image(img_path)

    print("✅ 所有图像识别完成。日志已生成。")
