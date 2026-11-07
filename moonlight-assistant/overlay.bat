@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
 py -3 -m venv .venv
 if errorlevel 1 goto failed
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
if "%~1"=="--calibrate" (
 .venv\Scripts\python.exe overlay.py %*
 if errorlevel 1 goto failed
 exit /b 0
)
if not exist live_config.json (
 echo Open the game in windowed or borderless mode before calibration.
 .venv\Scripts\python.exe overlay.py --calibrate
 if errorlevel 1 goto failed
)
if not exist live_config.json exit /b 0
.venv\Scripts\python.exe overlay.py %*
if errorlevel 1 goto failed
exit /b 0
:failed
echo Failed. Read the error above.
pause
exit /b 1
