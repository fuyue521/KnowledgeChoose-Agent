@echo off
REM ============================================
REM 一键启动：建索引 + 启动 Web 界面（Windows）
REM 前提：已 conda activate AI 并安装依赖
REM ============================================
chcp 65001 >nul
python build_index.py
if errorlevel 1 goto :fail
python app.py
goto :eof

:fail
echo.
echo [错误] 索引构建失败，请检查 data/raw 目录与 .env 配置。
pause
