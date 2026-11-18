@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  python start_person4.py %*
) else (
  py -3.12 start_person4.py %*
)
pause
