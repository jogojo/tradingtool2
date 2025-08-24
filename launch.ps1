# Script PowerShell pour lancer Trading Tool Pro
Write-Host "========================================" -ForegroundColor Green
Write-Host "    TRADING TOOL PRO - DEMARRAGE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

Write-Host "1. Test de l'application..." -ForegroundColor Yellow
& ".venv\Scripts\python.exe" "test_app.py"

Write-Host ""
Write-Host "2. Lancement de l'interface Streamlit..." -ForegroundColor Yellow
Write-Host "   L'application va s'ouvrir dans votre navigateur" -ForegroundColor Cyan
Write-Host "   Appuyez sur Ctrl+C pour arrêter" -ForegroundColor Cyan
Write-Host ""

& ".venv\Scripts\streamlit.exe" "run" "app.py"

Read-Host "Appuyez sur Entrée pour fermer"
