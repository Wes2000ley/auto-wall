@echo off
cd /d "%~dp0"
python fast_auto_wall.py %*
if errorlevel 1 pause
