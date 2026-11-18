@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  python start_person3.py %*
) else (
  py -3.12 start_person3.py %*
)
pause
