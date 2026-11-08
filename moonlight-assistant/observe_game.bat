@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe launch_neural.py --observe %*
if errorlevel 1 pause
