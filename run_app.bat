@echo off
call venv\Scripts\activate
python app.py
echo.
echo Exit code: %errorlevel%
cmd /k
