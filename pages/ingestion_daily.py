import streamlit as st
import os
from pathlib import Path


def render():
    """Affiche la page Ingestion Daily (EOD)"""
    st.header("🗓️ Ingestion Daily (EOD)")

    tab1, tab2 = st.tabs(["📁 Fichier Unique", "📂 Répertoire"])

    with tab1:
        st.subheader("Traitement d'un fichier Daily")
        uploaded_file = st.file_uploader(
            "Choisir un fichier (ZIP, CSV, TXT)",
            type=["zip", "csv", "txt"],
            key="daily_uploader_file"
        )

        if uploaded_file is not None:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            asset_type = st.selectbox(
                "Type d'asset",
                ["stock", "etf", "future", "crypto", "forex", "index"],
                key="daily_file_asset_type"
            )

            st.info(f"📁 Fichier : {uploaded_file.name}")
            st.info("🔍 Symbole : détection automatique depuis le nom du fichier")

            if st.button("Traiter le fichier Daily", key="daily_process_file_btn"):
                try:
                    with st.spinner("Traitement en cours..."):
                        results = st.session_state.data_ingestion_daily.process_file(temp_path, asset_type)
                    st.success("✅ Traitement Daily terminé !")
                    st.json(results)
                except Exception as e:
                    st.error(f"❌ Erreur Daily : {str(e)}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    with tab2:
        st.subheader("Traitement d'un répertoire Daily")
        source_dir = st.text_input("Chemin du répertoire source", key="daily_source_dir")

        if source_dir:
            asset_type_dir = st.selectbox(
                "Type d'asset",
                ["stock", "etf", "future", "crypto", "forex", "index"],
                key="daily_dir_asset_type"
            )

            if st.button("Traiter Daily", key="daily_process_dir_btn"):
                try:
                    source_path = Path(source_dir)
                    if not source_path.exists():
                        st.error("Répertoire non trouvé")
                        st.stop()

                    supported_files = list(source_path.glob("*.csv")) + list(source_path.glob("*.txt")) + list(source_path.glob("*.zip"))
                    if not supported_files:
                        st.warning("Aucun fichier Daily trouvé")
                        st.stop()

                    total_results = {"fichiers_traités": 0, "lignes_traitées": 0, "erreurs": 0}
                    progress = st.progress(0)

                    for idx, file in enumerate(supported_files):
                        st.write(f"Traitement de {file.name}...")
                        results = st.session_state.data_ingestion_daily.process_file(str(file), asset_type_dir)
                        total_results["fichiers_traités"] += results.get("fichiers_traités", 0)
                        total_results["lignes_traitées"] += results.get("lignes_traitées", 0)
                        total_results["erreurs"] += results.get("erreurs", 0)
                        progress.progress((idx + 1) / len(supported_files))

                    st.success(f"""Traitement Daily terminé !
                    - Fichiers traités : {total_results['fichiers_traités']}
                    - Lignes traitées : {total_results['lignes_traitées']}
                    - Erreurs : {total_results['erreurs']}""")
                except Exception as e:
                    st.error(f"Erreur Daily : {str(e)}")
