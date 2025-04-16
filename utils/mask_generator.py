import numpy as np
import cv2
import os

def generate_mask(image, detections, dilate_kernel_size=3, dilate_iter=1, save_path=None):
    """
    根据 OCR 检测结果生成掩码图像，并可选择保存调试用的掩码图

    参数:
        image (np.ndarray): 输入原图（BGR格式）
        detections (List[dict]): OCR识别结果，包含 'box'
        dilate_kernel_size (int): 膨胀核大小，默认3
        dilate_iter (int): 膨胀次数，默认1
        save_path (str): 可选，保存调试掩码图像的完整路径

    返回:
        mask (np.ndarray): 单通道掩码图，文字区域为255，其余为0
    """

    # 1. 创建全黑掩码图
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # 2. 绘制每个 OCR box
    for item in detections:
        box = np.array(item["box"], dtype=np.int32)
        cv2.fillPoly(mask, [box], 255)

    # 3. 膨胀处理
    if dilate_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    # 4. 可选保存掩码图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, mask)
        print(f"🧪 已保存调试掩码图至: {save_path}")

    return mask

