import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import os
import logging
import io
import sys
import time
import contextlib

# Import du module d'ingestion
from src.data.ingestion import DataIngestion

# Configuration du logging pour capturer les logs
class StreamlitHandler(logging.Handler):
    def __init__(self, streamlit_container):
        super().__init__()
        self.container = streamlit_container
        self.log_buffer = []
    
    def emit(self, record):
        log_entry = self.format(record)
        self.log_buffer.append(log_entry)
        
        # Afficher les logs dans Streamlit
        if self.container:
            with self.container:
                st.text_area("📋 Logs de traitement", "\n".join(self.log_buffer[-50:]), height=200)

# Configuration de la page
st.set_page_config(
    page_title="Trading Tool Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("📈 Trading Tool Pro")
st.markdown("---")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choisir une section",
    ["Dashboard", "Ingestion des Données", "Calendriers", "Analyses", "Configuration"]
)

# Initialiser l'ingestion des données
if 'data_ingestion' not in st.session_state:
    try:
        st.session_state.data_ingestion = DataIngestion()
    except Exception as e:
        st.error(f"Erreur d'initialisation : {e}")

# Dashboard principal
if page == "Dashboard":
    st.header("📊 Dashboard")
    
    # Récupérer les informations de stockage
    storage_info = st.session_state.data_ingestion.get_storage_info()
    
    # Calculer les totaux
    total_assets = sum(info['nombre_sous_jacents'] for info in storage_info.values())
    total_years = sum(info['années_total'] for info in storage_info.values())
    total_size = sum(info['taille_totale_gb'] for info in storage_info.values())
    
    # Métriques principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📈 Sous-jacents",
            value=total_assets,
            help="Nombre total de sous-jacents stockés"
        )
    
    with col2:
        st.metric(
            label="📅 Années de données",
            value=total_years,
            help="Nombre total d'années de données stockées"
        )
    
    with col3:
        st.metric(
            label="💾 Stockage total",
            value=f"{total_size:.2f} GB",
            help="Taille totale des données stockées"
        )
    
    # Détail par type d'asset
    st.subheader("📋 Détail par type d'asset")
    
    for asset_type, info in storage_info.items():
        if info['nombre_sous_jacents'] > 0:
            with st.expander(f"{asset_type.upper()} - {info['nombre_sous_jacents']} sous-jacents"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sous-jacents", info['nombre_sous_jacents'])
                with col2:
                    st.metric("Années", info['années_total'])
                with col3:
                    st.metric("Taille", f"{info['taille_totale_gb']} GB")
                
                if info['sous_jacents']:
                    st.write("**Sous-jacents :**", ", ".join(info['sous_jacents']))
                st.write(f"**Chemin :** `{info['chemin']}`")

elif page == "Ingestion des Données":
    # Section Ingestion des Données
    st.header("📊 Ingestion des Données")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Fichier Unique", "📂 Répertoire", "📊 Aperçu des Données", "📈 Lecture des Données"])
    
    with tab1:
        st.subheader("Traitement d'un fichier")
        uploaded_file = st.file_uploader(
            "Choisir un fichier (ZIP, CSV, TXT)", 
            type=['zip', 'csv', 'txt']
        )
        
        if uploaded_file is not None:
            # Sauvegarder le fichier temporairement
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            asset_type = st.selectbox(
                "Type d'asset",
                ["stock", "etf", "future", "crypto", "forex", "index"],
                key="file_asset_type"
            )
            
            st.info(f"📁 **Fichier :** {uploaded_file.name}")
            st.info(f"🔍 **Symbole :** Détection automatique depuis le nom du fichier")
            
            if st.button("Traiter le fichier"):
                try:
                    with st.spinner("Traitement en cours..."):
                        results = st.session_state.data_ingestion.process_file(temp_path, asset_type)
                    
                    st.success(f"✅ Traitement terminé !")
                    st.json(results)
                    
                    # Nettoyer le fichier temporaire
                    os.remove(temp_path)
                    
                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    with tab2:
        st.subheader("Traitement d'un répertoire")
        source_dir = st.text_input("Chemin du répertoire source")
        
        if source_dir:
            asset_type_dir = st.selectbox(
                "Type d'asset",
                ["stock", "etf", "future", "crypto", "forex", "index"],
                key="dir_asset_type"
            )
            
            # Un seul bouton pour tout faire
            if st.button("Traiter", key="single_process_btn"):
                try:
                    # Vérifier le répertoire
                    source_path = Path(source_dir)
                    if not source_path.exists():
                        st.error("Répertoire non trouvé")
                        st.stop()
                    
                    # Trouver les fichiers
                    supported_files = list(source_path.glob("*.csv")) + list(source_path.glob("*.txt")) + list(source_path.glob("*.zip"))
                    if not supported_files:
                        st.warning("Aucun fichier trouvé")
                        st.stop()
                    
                    # Traiter les fichiers
                    total_results = {"fichiers_traités": 0, "lignes_traitées": 0, "erreurs": 0}
                    progress = st.progress(0)
                    
                    for idx, file in enumerate(supported_files):
                        st.write(f"Traitement de {file.name}...")
                        results = st.session_state.data_ingestion.process_file(str(file), asset_type_dir)
                        
                        total_results["fichiers_traités"] += results.get("fichiers_traités", 0)
                        total_results["lignes_traitées"] += results.get("lignes_traitées", 0)
                        total_results["erreurs"] += results.get("erreurs", 0)
                        
                        progress.progress((idx + 1) / len(supported_files))
                    
                    st.success(f"""Traitement terminé !
                    - Fichiers traités : {total_results['fichiers_traités']}
                    - Lignes traitées : {total_results['lignes_traitées']}
                    - Erreurs : {total_results['erreurs']}""")
                    
                except Exception as e:
                    st.error(f"Erreur : {str(e)}")
        else:
            st.write("🔍 **DEBUG :** Chemin vide, pas d'affichage")
    
    with tab3:
        st.subheader("Informations de stockage")
        storage_info = st.session_state.data_ingestion.get_storage_info()
        
        for asset_type, info in storage_info.items():
            with st.expander(f"{asset_type.upper()} - {info['nombre_sous_jacents']} sous-jacents"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sous-jacents", info['nombre_sous_jacents'])
                with col2:
                    st.metric("Années", info['années_total'])
                with col3:
                    st.metric("Taille", f"{info['taille_totale_gb']} GB")
                
                if info['sous_jacents']:
                    st.write("**Sous-jacents :**", ", ".join(info['sous_jacents']))
                st.write(f"**Chemin :** `{info['chemin']}`")
    
    with tab4:
        st.subheader("Lecture des Données")
        
        # Sélection du type d'asset
        storage_info = st.session_state.data_ingestion.get_storage_info()
        asset_types_with_data = [
            asset_type for asset_type, info in storage_info.items() 
            if info['nombre_sous_jacents'] > 0
        ]
        
        if not asset_types_with_data:
            st.warning("Aucune donnée disponible. Veuillez d'abord ingérer des données.")
            st.stop()
            
        asset_type = st.selectbox(
            "Type d'asset",
            asset_types_with_data,
            key="read_asset_type"
        )
        
        # Sélection du symbole
        symbols = storage_info[asset_type]['sous_jacents']
        symbol = st.selectbox(
            "Symbole",
            symbols,
            key="read_symbol"
        )
        
        # Sélection de la période
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input(
                "Année de début",
                min_value=2000,
                max_value=2100,
                value=2020,
                key="read_start_year"
            )
        with col2:
            end_year = st.number_input(
                "Année de fin",
                min_value=2000,
                max_value=2100,
                value=2024,
                key="read_end_year"
            )
        
        if st.button("Charger les données"):
            try:
                with st.spinner("Chargement des données..."):
                    df = st.session_state.data_ingestion.read_symbol_data(
                        asset_type,
                        symbol,
                        start_year=start_year,
                        end_year=end_year
                    )
                
                st.success(f"✅ {len(df):,} lignes chargées")
                
                # Afficher les statistiques
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Première ligne :**")
                    st.write(df.iloc[0])
                with col2:
                    st.write("**Dernière ligne :**")
                    st.write(df.iloc[-1])
                
                # Afficher les données
                st.write("**Aperçu des données :**")
                st.dataframe(
                    df.head(1000),
                    use_container_width=True
                )
                
                # Statistiques
                st.write("**Statistiques :**")
                st.dataframe(
                    df.describe(),
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Erreur lors du chargement des données : {str(e)}")

elif page == "Calendriers":
    st.header("Calendriers de Trading")
    
    st.subheader("Configuration des Marchés")
    
    markets = ["US Stocks", "US Futures", "European Stocks", "Asian Stocks", "Forex", "Crypto"]
    
    for market in markets:
        with st.expander(f"📅 {market}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.time_input(f"Heure d'ouverture {market}", value=datetime.strptime("09:30", "%H:%M").time())
                st.time_input(f"Heure de fermeture {market}", value=datetime.strptime("16:00", "%H:%M").time())
            
            with col2:
                st.checkbox(f"RTH {market}")
                st.checkbox(f"ETH {market}")
                st.date_input(f"Date de début {market}", value=datetime(2000, 1, 1))

elif page == "Analyses":
    st.header("Analyses et Moyennes")
    
    st.subheader("Calcul des Moyennes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox("Période", ["20 ans", "10 ans", "5 ans", "1 an", "6 mois", "3 mois"])
        st.selectbox("Fréquence", ["1 minute", "5 minutes", "15 minutes", "1 heure", "Daily"])
        st.button("Calculer les moyennes", type="primary")
    
    with col2:
        st.selectbox("Sous-jacent", ["Sélectionner un sous-jacent"])
        st.date_input("Date de début")
        st.date_input("Date de fin")
    
    st.markdown("---")
    
    st.subheader("Résultats")
    st.info("Les moyennes seront calculées automatiquement à chaque mise à jour des données")

elif page == "Configuration":
    st.header("Configuration")
    
    st.subheader("Paramètres Généraux")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Répertoire de stockage", value="./data")
        st.number_input("Taille max cache (GB)", value=10, min_value=1, max_value=100)
        st.selectbox("Timezone", ["UTC", "Europe/Paris", "America/New_York"])
    
    with col2:
        st.text_input("Base de données", value="localhost:5432")
        st.text_input("Utilisateur DB")
        st.text_input("Mot de passe DB", type="password")
    
    st.markdown("---")
    
    st.subheader("First Rate Data")
    st.text_input("API Key", type="password")
    st.text_input("Endpoint API", value="https://api.firstratedata.com")
    
    st.markdown("---")
    
    st.subheader("IBKR")
    st.text_input("Port TWS", value="7497")
    st.text_input("Host TWS", value="localhost")
    st.checkbox("Connexion automatique")

# Footer
st.markdown("---")
st.markdown("**Trading Tool Pro** - Outil de trading professionnel")
st.markdown(f"*Dernière mise à jour : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
