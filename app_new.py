import streamlit as st
from datetime import datetime

# Import du module d'ingestion
from src.data.ingestion import DataIngestion
from src.data.ingestion_daily import DataIngestionDaily

# Import des pages modulaires
from pages import dashboard, average_day, calendars

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
    ["Dashboard", "Ingestion des Données", "Ingestion Daily (EOD)", "Silver - Fill Gaps", "Audit Silver", "Récup Prix", "Average Day", "Audit Bronze UTC", "Calendriers", "Analyses", "Configuration"]
)

# Initialiser l'ingestion des données
if 'data_ingestion' not in st.session_state:
    try:
        st.session_state.data_ingestion = DataIngestion()
    except Exception as e:
        st.error(f"Erreur d'initialisation : {e}")

# Initialiser l'ingestion daily
if 'data_ingestion_daily' not in st.session_state:
    try:
        st.session_state.data_ingestion_daily = DataIngestionDaily()
    except Exception as e:
        st.error(f"Erreur d'initialisation daily : {e}")

# Routing vers les pages modulaires
if page == "Dashboard":
    dashboard.render()

elif page == "Average Day":
    average_day.render()

elif page == "Calendriers":
    calendars.render()

# Pages temporaires (à refactoriser)
elif page == "Ingestion des Données":
    st.header("📊 Ingestion des Données")
    st.info("🚧 Page en cours de refactorisation...")

elif page == "Ingestion Daily (EOD)":
    st.header("🗓️ Ingestion Daily (EOD)")
    st.info("🚧 Page en cours de refactorisation...")

elif page == "Silver - Fill Gaps":
    st.header("🔄 Silver - Fill Gaps")
    st.info("🚧 Page en cours de refactorisation...")

elif page == "Audit Silver":
    st.header("🔎 Audit Silver")
    st.info("🚧 Page en cours de refactorisation...")

elif page == "Récup Prix":
    st.header("🎯 Récupération de prix")
    st.info("🚧 Page en cours de refactorisation...")

elif page == "Audit Bronze UTC":
    st.header("🔍 Audit Bronze UTC")
    st.info("🚧 Page en cours de refactorisation...")

elif page == "Analyses":
    st.header("📈 Analyses")
    st.info("🚧 Page en cours de développement...")

elif page == "Configuration":
    st.header("⚙️ Configuration")
    st.info("🚧 Page en cours de développement...")

# Footer
st.markdown("---")
st.markdown("**Trading Tool Pro** - Outil de trading professionnel")
st.markdown(f"*Dernière mise à jour : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
