import streamlit as st
import os
from pathlib import Path


def render():
    """Affiche la page Ingestion des Données"""
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
