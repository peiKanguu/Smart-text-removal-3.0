from paddleocr import PaddleOCR
import numpy as np

# 初始化 OCR 引擎（可重复调用，不会重复初始化）
ocr_engine = PaddleOCR(use_angle_cls=True, lang='ch')  # 支持中英文，带方向识别

def detect_text(image: np.ndarray):
    """
    对图像进行 OCR 文字检测与识别

    参数:
        image (np.ndarray): BGR 格式图像（OpenCV 读取）

    返回:
        List[Dict]: 每个检测文字的字典，包括坐标框、识别文字与置信度
    """
    if image is None:
        raise ValueError("图像为空，无法进行识别")

    # OCR 识别（PaddleOCR 默认要求 RGB）
    result = ocr_engine.ocr(image[:, :, ::-1], cls=True)

    detections = []
    for line in result:
        for box, (text, score) in line:
            detection = {
                "box": box,               # 四点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                "text": text,             # 识别出来的文字
                "score": float(score)     # 置信度
            }
            detections.append(detection)
    return detections
