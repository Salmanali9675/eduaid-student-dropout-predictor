@echo off
title Push EduAid to GitHub
echo ========================================================
echo Pushing EduAid Project to GitHub...
echo ========================================================
cd /d "%~dp0"
git push -u origin main
echo.
if %errorlevel% equ 0 (
    echo ========================================================
    echo SUCCESS! Your project is live on GitHub:
    echo https://github.com/Salmanali9675/eduaid-student-dropout-predictor
    echo ========================================================
) else (
    echo ========================================================
    echo If repository was not found, make sure you created
    echo 'eduaid-student-dropout-predictor' at:
    echo https://github.com/new
    echo ========================================================
)
pause
