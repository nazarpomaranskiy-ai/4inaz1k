@echo off
cd /d "%~dp0"
if "%~1"=="" (
 echo Drag a video onto this file, or run: run.bat video.mp4
 pause
 exit /b 1
)
if not exist .venv\Scripts\python.exe (
 py -3 -m venv .venv
 if errorlevel 1 goto failed
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe analyze.py "%~1" --output annotated.mp4
if errorlevel 1 goto failed
echo Finished. Open annotated.mp4 and annotated.csv.
pause
exit /b 0
:failed
echo Failed. Read the error above. Python 3.11 or 3.12 is recommended.
pause
exit /b 1
