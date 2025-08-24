@echo off
title Trading Tool Pro
echo ========================================
echo    TRADING TOOL PRO - DEMARRAGE
echo ========================================
echo.

echo Lancement de l'application...
echo.

REM Lancer Streamlit en arrière-plan
start /B .venv\Scripts\streamlit.exe run app.py --server.port 8501

REM Attendre que Streamlit soit prêt
echo Attente du démarrage de Streamlit...
timeout /t 5 /nobreak >nul

REM Ouvrir le navigateur
echo Ouverture du navigateur...
start http://localhost:8501

echo.
echo Application lancee sur http://localhost:8501
echo Appuyez sur une touche pour fermer cette fenetre
pause >nul
