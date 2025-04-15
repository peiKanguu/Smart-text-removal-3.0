import os
import cv2
import json
from tqdm import tqdm
from detect.detect_text import detect_text  # OCR 模块

# 配置路径
input_folder = './datasets/input_images'
output_log_folder = './outputs/detection_logs'

# 确保输出文件夹存在
os.makedirs(output_log_folder, exist_ok=True)

def process_image(img_path):
    img_name = os.path.basename(img_path)
    base_name = os.path.splitext(img_name)[0]

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 无法读取图像：{img_path}")
        return

    # 🧠 步骤 1：OCR识别
    detections = detect_text(img)  # list of dicts: {'text', 'score', 'bbox'}

    # 📝 步骤 2：写入日志 JSON 文件
    log_data = {
        "filename": img_name,
        "detections": []
    }

    for det in detections:
        log_data["detections"].append({
            "text": det["text"],
            "score": round(det["score"], 4),
            "bbox": det["bbox"]  # [x1, y1, x2, y2]
        })

    log_path = os.path.join(output_log_folder, base_name + '_ocr_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    print(f"✅ {img_name} -> 识别结果已保存至日志")

if __name__ == "__main__":
    print("🚀 开始批量处理图片...")
    for filename in tqdm(os.listdir(input_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            process_image(img_path)
    print("✅ 所有图像识别完成。日志已生成。")
