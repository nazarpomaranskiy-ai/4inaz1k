@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
 echo Run overlay.bat first to install the environment.
 pause
 exit /b 1
)
.venv\Scripts\python.exe review_learning.py
if errorlevel 1 pause
