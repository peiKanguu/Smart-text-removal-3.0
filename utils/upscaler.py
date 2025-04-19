# utils/upscaler.py

import os
import cv2
import subprocess
import sys
import torch

def upscale_with_realesrgan(img, base_name, output_folder='./outputs/super_resolution'):
    os.makedirs(output_folder, exist_ok=True)

    # 获取路径
    current_dir = os.path.dirname(os.path.abspath(__file__))  # utils/
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    realesrgan_path = os.path.join(project_root, 'third_party', 'Real-ESRGAN', 'inference_realesrgan.py')

    # 构建临时输入输出路径
    input_path = os.path.join(output_folder, f"{base_name}_temp_input.png")
    output_path = os.path.join(output_folder, f"{base_name}_temp_input_out.png")

    # 保存原图作为输入
    cv2.imwrite(input_path, img)

    # 检查 CUDA 可用性
    use_fp32 = not torch.cuda.is_available()
    if use_fp32:
        print("🖥️ 当前系统无可用 GPU，使用 FP32（CPU）模式进行超分")
    else:
        print("⚡ 检测到可用 GPU，使用默认 FP16 模式进行加速")

    # 获取当前 Python 路径
    python_executable = sys.executable

    # 构建命令行
    command = [
        python_executable, realesrgan_path,
        "-n", "RealESRGAN_x4plus",
        "-i", input_path,
        "-o", output_folder
    ]
    if use_fp32:
        command.append("--fp32")

    try:
        subprocess.run(command, check=True)
        result = cv2.imread(output_path)
        if result is None:
            raise ValueError("❌ 超分输出图像读取失败")
        print(f"🖼️ 超分图像已保存至：{output_path}")
        return result
    except Exception as e:
        print(f"❌ Real-ESRGAN 处理失败: {e}")
        return img
