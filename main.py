import os
import cv2
import json
from tqdm import tqdm
import numpy as np
np.int = int  # ✅ 解决 numpy.int 报错

# ✅ 检查 PyTorch 和 GPU 是否可用
try:
    import torch
    HAS_TORCH = True
    HAS_GPU = torch.cuda.is_available()
    print(f"🧠 PyTorch 检测成功，GPU 状态: {'可用' if HAS_GPU else '不可用'}")
except ImportError:
    HAS_TORCH = False
    HAS_GPU = False
    print("⚠️ 未安装 PyTorch，将使用 OpenCV 修复")

from utils.mask_generator import generate_mask
from utils.blur_detector import detect_blur_variance_laplacian
from utils.resolution_utils import is_low_resolution
from utils.image_scaler import enlarge_image
from utils.upscaler import upscale_with_realesrgan

# 配置路径
input_folder = './datasets/input_images'
output_log_folder = './outputs/detection_logs'
output_mask_folder = './outputs/mask_debug'
output_enlarge_folder = './outputs/enlarge_image'
output_superres_folder = './outputs/super_resolution'
os.makedirs(output_superres_folder, exist_ok=True)


# 安全导入 detect_text
try:
    from detect.detect_text import detect_text # OCR 模块
except Exception as e:
    print("❌ 导入模块失败:", e)
    exit(1)


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
    print()
    img_name = os.path.basename(img_path)
    base_name = os.path.splitext(img_name)[0]
    original_img = cv2.imread(img_path)

    if original_img is None:
        print(f"❌ 无法读取图像：{img_path}")
        return

    h, w = original_img.shape[:2]
    print(f"📐 当前图像分辨率：{w}x{h}")

    # 状态记录变量
    is_low_res = is_low_resolution(original_img)
    was_enlarged = False
    was_super_resolved = False
    was_modified = False
    working_img = original_img.copy()

    # ✅ 放大
    if is_low_res:
        print(f"📏 图像分辨率较低，执行插值放大")
        working_img = enlarge_image(working_img, scale=4)
        was_enlarged = True
        enlarged_path = os.path.join(output_enlarge_folder, base_name + '_enlarged.png')
        cv2.imwrite(enlarged_path, working_img)



    # ✅ OCR识别
    detections = detect_text(working_img)
    print(f"📄 正在处理：{img_name} - 🔍 识别到 {len(detections)} 个文字区域")

    if detections:
        for i, item in enumerate(detections, 1):
            print(f"   ✏️ [{i}] \"{item['text']}\"（置信度: {item['score']:.2f}）")
        was_modified = True
    else:
        print("⚠️ 未检测到文字，直接保存原图为清洁图")
        output_cleaned_path = os.path.join('./outputs/cleaned_images', base_name + '_cleaned.png')
        cv2.imwrite(output_cleaned_path, original_img)
        # ✅ 写日志
        log_data = {
            "filename": img_name,
            "original_resolution": f"{w}x{h}",
            "is_low_resolution": is_low_res,
            "was_enlarged": was_enlarged,
            "was_super_resolved": was_super_resolved,
            "was_modified": was_modified,
            "detections": []
        }
        log_path = os.path.join(output_log_folder, base_name + '_ocr_log.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        print(f"📄 日志记录已保存：{log_path}")
        return  # ✅ 跳过后续步骤

    # ✅ 掩码生成 + 修复
    mask_path = os.path.join(output_mask_folder, base_name + '_mask.png')
    mask = generate_mask(working_img, detections, save_path=mask_path)

    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        print(f"❌ 无法读取掩码图像：{mask_path}")
        return

#    method = choose_inpaint_method(working_img, mask_gray)
#    inpainted = cv2.inpaint(working_img, mask_gray, inpaintRadius=3, flags=method)
    # === 智能选择修复方式（优先使用 LaMa） ===
    lama_output_path = os.path.join('./outputs/cleaned_images', base_name + '_cleaned.png')

    use_lama = HAS_TORCH and HAS_GPU  # 如果你希望 CPU 也支持 LaMa 可放宽条件

    if use_lama:
        try:
            print("🧠 使用 LaMa 进行大模型修复")
            from lama_cleaner_model.run_lama_cleaner import run_lama_cleaner
            
            # 如果 working_img 是经过放大/超分的，需要先保存临时图
            working_img_path = os.path.join('./outputs/tmp', base_name + '_working.png')
            os.makedirs('./outputs/tmp', exist_ok=True)
            cv2.imwrite(working_img_path, working_img)
            
            run_lama_cleaner(
                image_path=working_img_path,
                mask_path=mask_path,
                output_path=lama_output_path
            )

            inpainted = cv2.imread(lama_output_path)
            if inpainted is None:
                raise ValueError("LaMa 修复失败，未能读取修复结果图像")

        except Exception as e:
            print(f"❌ LaMa 修复失败，回退 OpenCV: {e}")
            method = choose_inpaint_method(working_img, mask_gray)
            inpainted = cv2.inpaint(working_img, mask_gray, inpaintRadius=3, flags=method)

    else:
        print("🧩 未检测到 PyTorch + GPU，使用 OpenCV 修复")
        method = choose_inpaint_method(working_img, mask_gray)
        inpainted = cv2.inpaint(working_img, mask_gray, inpaintRadius=3, flags=method)


    # ✅ 恢复输出为原图大小
    final_output = cv2.resize(inpainted, (w, h))
    output_cleaned_path = os.path.join('./outputs/cleaned_images', base_name + '_cleaned.png')
    cv2.imwrite(output_cleaned_path, final_output)
    print(f"🖼️ 图像修复完成：{output_cleaned_path}")

    # ✅ 写日志
    log_data = {
        "filename": img_name,
        "original_resolution": f"{w}x{h}",
        "is_low_resolution": is_low_res,
        "was_enlarged": was_enlarged,
        "was_super_resolved": was_super_resolved,
        "was_modified": was_modified,
        "detections": [  # 仅当 modified 为 True 时有内容
            {
                "text": d.get("text", ""),
                "score": round(d.get("score", 0.0), 4),
                "bbox": d.get("box", [])
            }
            for d in detections
        ]
    }

    log_path = os.path.join(output_log_folder, base_name + '_ocr_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"📄 日志记录已保存：{log_path}")


if __name__ == "__main__":
    print("🚀 开始批量处理图片...")
    # ✅ 获取所有待处理图片路径
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    print(f"📸 待处理图片数量: {len(image_files)}")

    # ✅ 使用 tqdm 显示处理进度
    for filename in tqdm(image_files, desc="处理中"):
        img_path = os.path.join(input_folder, filename)
        process_image(img_path)

    print("✅ 所有图像识别完成。日志已生成。")
