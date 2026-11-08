@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe launch_neural.py %*
if errorlevel 1 pause
