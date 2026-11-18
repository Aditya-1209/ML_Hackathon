@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  python start_person2.py %*
) else (
  py -3.12 start_person2.py %*
)
pause
