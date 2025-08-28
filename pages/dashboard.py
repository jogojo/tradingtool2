import streamlit as st
from src.data.ingestion import DataIngestion


def render():
    """Affiche la page Dashboard"""
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
