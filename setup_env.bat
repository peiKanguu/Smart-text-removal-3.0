@echo off
chcp 65001 >nul
echo 检测是否有 NVIDIA GPU...

REM ====== 检测 GPU 状态 ======
nvidia-smi >nul 2>nul
SET GPU_AVAILABLE=0

IF %ERRORLEVEL% EQU 0 (
    echo ✅ 检测到 NVIDIA GPU
    SET GPU_AVAILABLE=1
) ELSE (
    echo ⚠️ 未检测到 NVIDIA GPU
)

REM ====== 创建并激活虚拟环境 ======
python -m venv lama_env
call lama_env\Scripts\activate

REM ====== 安装 PyTorch 版本 ======
IF %GPU_AVAILABLE%==1 (
    echo 🚀 安装 GPU 版本 torch + torchvision（cu118）
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
) ELSE (
    echo 🧱 安装 CPU 版本 torch + torchvision
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.1
)

REM ====== 安装其他依赖 ======
echo 📦 安装 requirements.txt 中的依赖...
pip install -r requirements.txt

echo ✅ 虚拟环境与依赖安装完成！
pause
