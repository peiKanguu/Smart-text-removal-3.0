import cv2
import numpy as np

def detect_blur_variance_laplacian(img, threshold=100.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    var = laplacian.var()
    is_blur = var < threshold
    return {
        "is_blur": is_blur,
        "method": "variance_laplacian",
        "score": var
    }

def is_image_blurry(img):
    """
    后期可支持多种方法，如 FFT、Sobel、Tenengrad 等
    返回:
        {
            "is_blur": True/False,
            "method": "variance_laplacian",
            "score": float
        }
    """
    is_blur, score = detect_blur_variance_laplacian(img)
    return {
        "is_blur": is_blur,
        "method": "variance_laplacian",
        "score": score
    }
