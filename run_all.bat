@echo off
chcp 65001 >nul

REM 设置控制台窗口标题
title Smart Text Removal 3.0

REM 进入当前脚本所在目录（避免路径错误）
cd /d %~dp0

REM 激活虚拟环境
call lama_env\Scripts\activate

echo 🔄 虚拟环境已激活，正在启动 main.py...
echo --------------------------------------------

REM 检查 detect_text.py 是否存在
if not exist detect\detect_text.py (
    echo ❌ 错误：找不到 detect\detect_text.py 文件！
    pause
    exit /b
)

REM 启动主程序
python main.py

echo --------------------------------------------
echo ✅ 程序执行完毕！
pause
