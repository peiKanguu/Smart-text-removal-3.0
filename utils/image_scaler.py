import cv2

def enlarge_image(img, scale=2, method="cubic"):
    """
    简单图像放大函数（无细节增强）

    参数:
        img: 输入图像（np.ndarray）
        scale: 放大倍数
        method: 插值方式，支持 "nearest", "linear", "cubic", "lanczos"

    返回:
        放大后的图像（np.ndarray）
    """
    methods = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    h, w = img.shape[:2]
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=methods[method])
