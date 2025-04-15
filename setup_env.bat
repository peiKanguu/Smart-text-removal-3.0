@echo off
chcp 65001 >nul

REM 创建虚拟环境 lama_env（如果不存在）
python -m venv lama_env

REM 激活虚拟环境
call lama_env\Scripts\activate

REM 安装依赖
pip install -r requirements.txt

echo ✅ 环境创建成功，依赖安装完成！
pause
