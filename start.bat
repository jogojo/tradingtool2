@echo off
title Trading Tool Pro
echo ========================================
echo    TRADING TOOL PRO - DEMARRAGE
echo ========================================
echo.

echo 1. Test de l'application...
python test_app.py

echo.
echo 2. Lancement de l'interface...
echo    L'application va s'ouvrir dans votre navigateur
echo    Appuyez sur Ctrl+C pour arrêter
echo.

streamlit run app.py

pause
