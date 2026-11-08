@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
 py -3.12 -m venv .venv
 if errorlevel 1 goto failed
)
.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt -r requirements-extract.txt
if errorlevel 1 goto failed
set "ML_RESOURCES=%LOCALAPPDATA%\RealmOfTheMadGod\Production\RotMG Exalt_Data\resources.assets"
if not "%~1"=="" set "ML_RESOURCES=%~1"
if not exist "%ML_RESOURCES%" goto failed
.venv\Scripts\python.exe extract_assets.py --resources "%ML_RESOURCES%"
if errorlevel 1 goto failed
echo Reference ready. overlay.bat is advisory only. Motion requires a labeled replay check.
pause
exit /b 0
:failed
echo Setup failed. Read the error above.
pause
exit /b 1
