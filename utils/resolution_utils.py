def is_low_resolution(img, threshold=128*128):
    h, w = img.shape[:2]
    return (h * w) < threshold
