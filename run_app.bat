@echo off
title EduAid AI - Student Dropout Prevention System
echo ========================================================
echo Starting EduAid AI Early-Warning Web Dashboard...
echo ========================================================
cd /d "%~dp0"
py -m streamlit run app.py
pause
