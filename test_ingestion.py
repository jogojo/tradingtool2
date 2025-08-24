#!/usr/bin/env python3
"""
Script de test simple pour vérifier le module d'ingestion
"""

import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    print("🔍 Test d'import du module d'ingestion...")
    from src.data.ingestion import DataIngestion
    print("✅ Import réussi !")
    
    print("\n🔍 Création de l'instance DataIngestion...")
    ingestion = DataIngestion()
    print(f"✅ Instance créée : {type(ingestion)}")
    print(f"🔍 Répertoire de base : {ingestion.base_data_dir}")
    
    print("\n🔍 Test de détection de symbole...")
    test_files = [
        "EURAUD_full_1min.txt",
        "GBPCHF_full_1min.txt", 
        "GBPUSD_full_1min.txt"
    ]
    
    for test_file in test_files:
        symbol = ingestion._detect_symbol_from_path(test_file, "forex")
        print(f"📁 {test_file} → Symbole détecté : {symbol}")
    
    print("\n🔍 Test de traitement d'un fichier...")
    # Créer un fichier de test simple
    test_csv = "test_sample.csv"
    with open(test_csv, "w") as f:
        f.write("timestamp,open,high,low,close\n")
        f.write("2025-08-24 09:30:00,1.2000,1.2010,1.1990,1.2005\n")
        f.write("2025-08-24 09:31:00,1.2005,1.2020,1.2000,1.2015\n")
    
    print(f"📄 Fichier de test créé : {test_csv}")
    
    print("\n🔍 Lancement du traitement...")
    results = ingestion.process_file(test_csv, "forex")
    print(f"✅ Résultats : {results}")
    
    # Nettoyer
    import os
    os.remove(test_csv)
    print(f"🧹 Fichier de test supprimé")
    
except Exception as e:
    print(f"❌ ERREUR : {e}")
    import traceback
    traceback.print_exc()
