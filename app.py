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
import pyarrow.dataset as ds

# Import du module d'ingestion
from src.data.ingestion import DataIngestion
from src.data.ingestion_daily import DataIngestionDaily
from src.calendar.session_loader import TradingSessionTemplates, SymbolSessionRegistry
from src.silver.run_fill import run_gap_fill_for_symbol
from src.tools.audit_bronze_utc import audit_bronze
from src.tools.audit_silver import audit_silver as audit_silver_tool
from src.tools.fetch_prices import fetch_prices, fetch_prices_at_hhmm
from src.tools.average_day import compute_average_day, plot_average_day

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

elif page == "Ingestion Daily (EOD)":
    # Section Ingestion Daily (EOD)
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

elif page == "Calendriers":
    st.header("🗓️ Sessions & Mapping")

    # Templates
    templates = TradingSessionTemplates()
    with st.expander("📚 Templates de sessions", expanded=True):
        st.json(templates.templates)

    # Auto-mapping par répertoire
    with st.expander("🤖 Auto-mapping par répertoire", expanded=True):
        reg = SymbolSessionRegistry()
        
        # Répertoire à scanner
        base_dir = st.text_input("📁 Chemin du répertoire à scanner", value="data", key="auto_base_dir")
        
        # Template à appliquer
        template_to_apply = st.selectbox(
            "Template à appliquer", 
            list(templates.templates.keys()), 
            key="auto_template"
        )
        
        if st.button("🚀 Mapper tous les symboles", type="primary"):
            try:
                with st.spinner("Scan..."):
                    base_path = Path(base_dir)
                    if not base_path.exists():
                        st.error(f"Répertoire introuvable: {base_path}")
                        st.stop()
                    
                    mapped_count = 0
                    total_count = 0
                    details = []
                    
                    def extract_symbol_from_filename(filename: str) -> str:
                        """Extrait le symbole du nom de fichier (avant le premier _)"""
                        name = filename.split('.')[0]  # Enlever l'extension
                        return name.split('_')[0]  # Prendre la partie avant le premier _
                    
                    # Cas 1: Structure organisée (bronze/, silver/, daily/)
                    has_organized_structure = False
                    for tier in ["bronze", "silver", "daily"]:
                        tier_dir = base_path / tier
                        if tier_dir.exists():
                            has_organized_structure = True
                            # Tous les asset_class=*
                            for asset_dir in tier_dir.iterdir():
                                if not (asset_dir.is_dir() and asset_dir.name.startswith("asset_class=")):
                                    continue
                                    
                                # Tous les symbol=*
                                for symbol_dir in asset_dir.iterdir():
                                    if symbol_dir.is_dir() and symbol_dir.name.startswith("symbol="):
                                        symbol = symbol_dir.name.replace("symbol=", "")
                                        total_count += 1
                                        
                                        # Si pas déjà mappé, appliquer le template
                                        if symbol.upper() not in reg.map:
                                            reg.set(symbol, template_to_apply)
                                            mapped_count += 1
                                            details.append(f"{symbol} → {template_to_apply}")
                    
                    # Cas 2: Fichiers directement dans le répertoire
                    if not has_organized_structure:
                        # Scanner les fichiers CSV/TXT/ZIP
                        file_patterns = ["*.csv", "*.txt", "*.zip"]
                        for pattern in file_patterns:
                            for file_path in base_path.glob(pattern):
                                symbol = extract_symbol_from_filename(file_path.name)
                                total_count += 1
                                
                                # Si pas déjà mappé, appliquer le template
                                if symbol.upper() not in reg.map:
                                    reg.set(symbol, template_to_apply)
                                    mapped_count += 1
                                    details.append(f"{symbol} → {template_to_apply}")
                
                st.success(f"✅ Mappé {mapped_count} nouveaux symboles sur {total_count} trouvés")
                
                if details:
                    with st.expander("Détails"):
                        for detail in details[:50]:
                            st.write(f"• {detail}")
                        if len(details) > 50:
                            st.write(f"... et {len(details) - 50} autres")
                
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
    
    # Affichage du mapping actuel
    with st.expander("📋 Mapping actuel", expanded=False):
        reg = SymbolSessionRegistry()
        if reg.map:
            # Statistiques
            stats = reg.get_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total symboles", stats["total_symbols"])
            with col2:
                st.write("**Par template:**")
                for template, count in stats["templates_used"].items():
                    st.write(f"• {template}: {count}")
            
            # Recherche dans le mapping
            search_mapping = st.text_input("🔍 Filtrer mapping", placeholder="Ex: AUD, BTC, equity...", key="search_mapping")
            
            # Affichage filtré
            mappings_to_show = reg.map.copy()
            if search_mapping:
                search_lower = search_mapping.lower()
                mappings_to_show = {
                    symbol: template for symbol, template in reg.map.items()
                    if search_lower in symbol.lower() or search_lower in template.lower()
                }
            
            if mappings_to_show:
                # Limiter l'affichage pour performance
                items_to_show = list(mappings_to_show.items())[:100]
                
                st.write(f"**Mapping ({len(items_to_show)} affichés sur {len(mappings_to_show)} trouvés):**")
                
                # Tableau avec colonnes
                for i in range(0, len(items_to_show), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(items_to_show):
                            symbol, template = items_to_show[i + j]
                            with col:
                                st.write(f"**{symbol}** → `{template}`")
                
                if len(mappings_to_show) > 100:
                    st.info(f"💡 Trop de résultats. Utilisez la recherche pour affiner ({len(mappings_to_show)} au total).")
                    
                # Export CSV
                if st.button("📥 Exporter mapping (CSV)"):
                    import csv
                    import io
                    
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow(["symbol", "template"])
                    for symbol, template in reg.map.items():
                        writer.writerow([symbol, template])
                    
                    st.download_button(
                        label="💾 Télécharger mapping.csv",
                        data=output.getvalue(),
                        file_name="symbol_mapping.csv",
                        mime="text/csv"
                    )
            else:
                if search_mapping:
                    st.info("Aucun résultat pour cette recherche.")
                else:
                    st.info("Pas de mapping trouvé.")
        else:
            st.info("Aucun mapping configuré. Utilisez l'auto-mapping ci-dessus ou la gestion individuelle ci-dessous.")

    # Recherche et gestion individuelle
    with st.expander("🔍 Recherche & mapping individuel", expanded=True):
        reg = SymbolSessionRegistry()
        stats = reg.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Symboles mappés", stats["total_symbols"])
        with col2:
            st.write("**Templates utilisés:**")
            for tpl, count in stats["templates_used"].items():
                st.write(f"• {tpl}: {count}")
        
        # Recherche
        search_query = st.text_input("🔍 Rechercher symboles", placeholder="Ex: AAPL, EUR, BTC...", key="symbol_search")
        
        if search_query:
            matches = reg.search_symbols(search_query, limit=50)
            if matches:
                st.write(f"**{len(matches)} résultat(s) trouvé(s):**")
                
                # Édition en ligne
                for i, match in enumerate(matches):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"**{match['symbol']}**")
                    with col2:
                        new_template = st.selectbox(
                            "Template", 
                            list(templates.templates.keys()), 
                            index=list(templates.templates.keys()).index(match['template']) if match['template'] in templates.templates else 0,
                            key=f"edit_{match['symbol']}_{i}"
                        )
                        if new_template != match['template']:
                            reg.set(match['symbol'], new_template)
                            st.rerun()
                    with col3:
                        if st.button("🗑️", key=f"del_{match['symbol']}_{i}", help="Supprimer"):
                            reg.map.pop(match['symbol'].upper(), None)
                            reg._save()
                            st.rerun()
            else:
                st.info("Aucun résultat trouvé")
        
        # Import CSV
        st.write("**Import CSV (symbol,template):**")
        csv_map = st.file_uploader("Fichier CSV", type=["csv"], key="map_csv")
        if csv_map and st.button("Importer CSV"):
            import io as _io
            import csv
            content = _io.StringIO(csv_map.getvalue().decode("utf-8"))
            reader = csv.reader(content)
            cnt = 0
            for row in reader:
                if len(row) >= 2:
                    reg.set(row[0], row[1])
                    cnt += 1
            st.success(f"{cnt} mappings importés")
            st.rerun()

elif page == "Silver - Fill Gaps":
    st.header("💿 Silver: Fill Gaps")

    def discover_symbols_for(asset_type: str) -> list:
        symbols = set()
        base = Path("data")
        for tier in ["bronze", "daily"]:
            d = base / tier / f"asset_class={asset_type}"
            if d.exists():
                for sd in d.iterdir():
                    if sd.is_dir() and sd.name.startswith("symbol="):
                        symbols.add(sd.name.replace("symbol=", ""))
        return sorted(symbols)

    templates = TradingSessionTemplates()
    with st.expander("▶️ Lancer", expanded=True):
        asset_fg = st.selectbox("Type d'asset", ["stock", "etf", "future", "crypto", "forex", "index"], key="fg_asset")
        all_syms = discover_symbols_for(asset_fg)
        sel_syms = st.multiselect("Symboles (multi)", all_syms, default=all_syms[:1], key="fg_symbols")
        reg = SymbolSessionRegistry()
        tpl_fg = st.selectbox("Template de session (défaut si non mappé)", list(templates.templates.keys()), index=0)
        col1, col2 = st.columns(2)
        with col1:
            sd = st.date_input("Date début (optionnel)", value=None)
        with col2:
            ed = st.date_input("Date fin (optionnel)", value=None)

        # Aperçu compteurs d'années attendues (daily) vs déjà remplies (silver)
        if sel_syms:
            stats = []
            for s in sel_syms:
                daily_dir = Path("data") / "daily" / f"asset_class={asset_fg}" / f"symbol={s}"
                silver_dir = Path("data") / "silver" / f"asset_class={asset_fg}" / f"symbol={s}"
                years = 0
                years_silver = 0
                if daily_dir.exists():
                    dset = ds.dataset(daily_dir, format="parquet")
                    tbl = dset.to_table()
                    df = tbl.to_pandas()
                    if not df.empty:
                        if sd:
                            df = df[df["date"] >= pd.to_datetime(sd).date()]
                        if ed:
                            df = df[df["date"] <= pd.to_datetime(ed).date()]
                        years = len(pd.Series(df["date"]).apply(lambda d: d.year).unique())
                # Compter les années déjà écrites en silver (par répertoires year=YYYY)
                if silver_dir.exists():
                    years_silver = len([d for d in silver_dir.iterdir() if d.is_dir() and d.name.startswith("year=")])
                stats.append({
                    "symbole": s,
                    "années_attendues_daily": years,
                    "années_deja_filled": years_silver,
                    "template": reg.get(s) or tpl_fg
                })
            df_stats = pd.DataFrame(stats)
            st.write("Aperçu années par symbole (attendues vs déjà filled):")
            st.dataframe(df_stats, use_container_width=True)
            # Résumé total
            st.info(f"Total années attendues: {int(df_stats['années_attendues_daily'].sum())} | Déjà filled: {int(df_stats['années_deja_filled'].sum())}")

        if st.button("▶️ Exécuter Fill Gaps (multi)") and sel_syms:
            progress = st.progress(0)
            summary = {"symbols_processed": 0, "details": []}
            total_expected_years = 0
            total_filled_years_before = 0
            # Snapshot avant exécution
            pre = {}
            for s in sel_syms:
                silver_dir = Path("data") / "silver" / f"asset_class={asset_fg}" / f"symbol={s}"
                years_silver = len([d for d in silver_dir.iterdir() if d.is_dir() and d.name.startswith("year=")]) if silver_dir.exists() else 0
                pre[s] = years_silver
            for i, s in enumerate(sel_syms):
                try:
                    res = run_gap_fill_for_symbol(
                        base_data_dir="data",
                        asset_type=asset_fg,
                        symbol=s,
                        session_template=reg.get(s) or tpl_fg,
                        start_date=sd.isoformat() if sd else None,
                        end_date=ed.isoformat() if ed else None,
                    )
                    # recompute years for reporting
                    daily_dir = Path("data") / "daily" / f"asset_class={asset_fg}" / f"symbol={s}"
                    years = 0
                    if daily_dir.exists():
                        dset = ds.dataset(daily_dir, format="parquet")
                        df = dset.to_table().to_pandas()
                        if not df.empty:
                            if sd:
                                df = df[df["date"] >= pd.to_datetime(sd).date()]
                            if ed:
                                df = df[df["date"] <= pd.to_datetime(ed).date()]
                            years = len(pd.Series(df["date"]).apply(lambda d: d.year).unique())
                    total_expected_years += years
                    summary["details"].append({
                        "symbole": s,
                        "années_daily": years,
                        "processed_days": res.get("processed_days", 0),
                        "written_files": res.get("written_files", 0)
                    })
                    summary["symbols_processed"] += 1
                except Exception as e:
                    summary["details"].append({"symbole": s, "erreur": str(e)})
                progress.progress((i + 1) / len(sel_syms))
            st.success("Remplissage terminé")
            st.json(summary)
            st.info(f"Symboles traités: {summary['symbols_processed']}")
            # Résumé après exécution
            post_filled = 0
            for s in sel_syms:
                silver_dir = Path("data") / "silver" / f"asset_class={asset_fg}" / f"symbol={s}"
                years_silver = len([d for d in silver_dir.iterdir() if d.is_dir() and d.name.startswith("year=")]) if silver_dir.exists() else 0
                post_filled += years_silver
            pre_filled = sum(pre.values())
            st.info(f"Années attendues: {total_expected_years} | Déjà filled avant: {pre_filled} | Après: {post_filled}")

elif page == "Audit Bronze UTC":
    st.header("🔍 Audit Bronze: Timestamps en UTC")
    st.write("Vérifie que tous les fichiers Parquet de la Bronze ont une colonne 'timestamp' tz-aware en UTC.")

    base_dir = st.text_input("Répertoire des données", value="./data", key="audit_base")
    verbose = st.checkbox("Mode verbeux (afficher chaque fichier)", value=False, key="audit_verbose")

    if st.button("Lancer l'audit", type="primary"):
        try:
            with st.spinner("Audit en cours..."):
                results, issues_summary = audit_bronze(Path(base_dir))

            total = len(results)
            ok = sum(1 for r in results if r.get("arrow_ok") and not r.get("issues"))
            to_fix = total - ok

            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("Fichiers scannés", total)
            with colB:
                st.metric("Fichiers OK", ok)
            with colC:
                st.metric("À corriger", to_fix)

            if issues_summary:
                st.subheader("Problèmes détectés")
                st.dataframe(
                    pd.DataFrame(
                        [{"issue": k, "count": v} for k, v in sorted(issues_summary.items(), key=lambda kv: (-kv[1], kv[0]))]
                    ),
                    use_container_width=True
                )

            if verbose:
                st.subheader("Détails par fichier")
                st.dataframe(pd.DataFrame(results), use_container_width=True)

            st.success("Audit terminé")
        except Exception as e:
            st.error(f"Erreur durant l'audit: {e}")

elif page == "Récup Prix":
    st.header("🎯 Récupération de prix (Silver)")
    st.write("Récupère les OHLCV aux minutes exactes renseignées.")

    def _discover_silver_symbols(asset_type: str) -> list:
        symbols = set()
        base = Path("data") / "silver" / f"asset_class={asset_type}"
        if base.exists():
            for sd in base.iterdir():
                if sd.is_dir() and sd.name.startswith("symbol="):
                    symbols.add(sd.name.replace("symbol=", ""))
        return sorted(symbols)

    asset = st.selectbox("Type d'asset", ["stock", "etf", "future", "crypto", "forex", "index"], key="px_asset")
    syms = _discover_silver_symbols(asset)
    if not syms:
        st.warning("Aucune donnée Silver disponible pour ce type d'asset.")
        st.stop()
    sym = st.selectbox("Symbole", syms, key="px_symbol")

    st.write("Saisir: soit des timestamps UTC (une par ligne), soit une heure 'HH:MM' (ex: 23:03 ou 23.03). Choisir la timezone pour HH:MM.")
    txt = st.text_area("Entrées", height=160, key="px_ts")
    tz_options = [
        "UTC",
        "Europe/London", 
        "Europe/Paris",
        "America/New_York",
        "Asia/Shanghai"
    ]
    tz_choice = st.selectbox("Timezone pour HH:MM", tz_options, index=0, key="px_tz")

    if st.button("Récupérer"):
        try:
            raw_lines = [l.strip() for l in txt.splitlines() if l.strip()]
            if not raw_lines:
                st.warning("Aucun timestamp fourni.")
                st.stop()

            # Si une seule ligne au format HH:MM → recherche sur toute l'historique
            if len(raw_lines) == 1:
                cand = raw_lines[0].replace(".", ":")
                if len(cand) == 5 and cand[2] == ":":
                    out = fetch_prices_at_hhmm(Path("data"), asset, sym, cand, timezone=tz_choice)
                else:
                    ts_parse = [pd.to_datetime(cand, utc=True, errors="raise")]
                    out = fetch_prices(Path("data"), asset, sym, ts_parse)
            else:
                # Sinon, parse en UTC (lignes ISO) et fetch exact
                ts_parse = [pd.to_datetime(l, utc=True, errors="raise") for l in raw_lines]
                out = fetch_prices(Path("data"), asset, sym, ts_parse)

            st.success(f"Trouvé {len(out)} minute(s)")
            # Séparer date et heure localisées selon tz_choice + UTC
            df_show = out.copy()
            if not df_show.empty:
                ts_utc = pd.to_datetime(df_show["timestamp"], utc=True)
                ts_local = ts_utc.dt.tz_convert(tz_choice)
                
                df_show["date"] = ts_local.dt.date.astype(str)
                df_show["heure"] = ts_local.dt.strftime("%H:%M")
                df_show["heure_utc"] = ts_utc.dt.strftime("%H:%M")
                
                # Ordonner colonnes: date, heure (locale), heure_utc, ohlcv, filled_from_ts
                cols = [c for c in ["date","heure","heure_utc","open","high","low","close","volume","filled_from_ts"] if c in df_show.columns]
                df_show = df_show[cols]
            st.dataframe(df_show, use_container_width=True)

            # téléchargement CSV
            csv = df_show.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger CSV", data=csv, file_name=f"{asset}_{sym}_prices.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Erreur: {e}")

elif page == "Average Day":
    st.header("📊 Average Day")
    st.write("Calcul du profil intraday moyen pour un sous-jacent (timezone locale + session).")

    def _discover_silver_symbols_avg(asset_type: str) -> list:
        symbols = set()
        base = Path("data") / "silver" / f"asset_class={asset_type}"
        if base.exists():
            for sd in base.iterdir():
                if sd.is_dir() and sd.name.startswith("symbol="):
                    symbols.add(sd.name.replace("symbol=", ""))
        return sorted(symbols)

    asset = st.selectbox("Type d'asset", ["stock", "etf", "future", "crypto", "forex", "index"], key="avg_asset")
    
    # Interface de recherche pour 5000+ symboles
    search_symbol = st.text_input("🔍 Rechercher symbole", placeholder="Ex: AAPL, MSFT, EUR...", key="avg_search")
    
    if search_symbol:
        syms = _discover_silver_symbols_avg(asset)
        # Filtrer par recherche
        search_lower = search_symbol.lower()
        filtered_syms = [s for s in syms if search_lower in s.lower()][:50]  # Limiter à 50
        
        if not filtered_syms:
            st.warning(f"Aucun symbole trouvé pour '{search_symbol}' dans {asset}")
            st.stop()
        
        if len(filtered_syms) == 1:
            sym = filtered_syms[0]
            st.success(f"Symbole sélectionné: **{sym}**")
        else:
            st.write(f"**{len(filtered_syms)} résultat(s) trouvé(s):**")
            sym = st.selectbox("Choisir le symbole", filtered_syms, key="avg_symbol_filtered")
    else:
        # Affichage classique mais limité
        syms = _discover_silver_symbols_avg(asset)[:100]  # Limiter à 100 pour la performance
        if not syms:
            st.warning("Aucune donnée Silver disponible pour ce type d'asset.")
            st.stop()
        sym = st.selectbox("Symbole (100 premiers)", syms, key="avg_symbol")
        if len(_discover_silver_symbols_avg(asset)) > 100:
            st.info("💡 Plus de 100 symboles disponibles. Utilisez la recherche ci-dessus pour plus de choix.")
    
    price_col = st.selectbox("Colonne de prix", ["close", "open", "high", "low"], index=0, key="avg_price")

    if st.button("Calculer Average Day", type="primary"):
        try:
            with st.spinner("Calcul de l'average day..."):
                df_avg, metadata = compute_average_day(Path("data"), asset, sym, price_col)

            if df_avg.empty:
                st.error(f"Erreur: {metadata.get('error', 'Aucune donnée')}")
                st.stop()

            # Statistiques compactes + min/max et pourcentage annualisé
            try:
                price_max = float(df_avg["avg_price"].max()) if not df_avg.empty else None
                price_min = float(df_avg["avg_price"].min()) if not df_avg.empty else None
                # Heures associées
                max_hhmm = None
                min_hhmm = None
                if price_max is not None and not df_avg.empty:
                    idxmax = df_avg["avg_price"].idxmax()
                    if idxmax is not None and idxmax in df_avg.index:
                        max_hhmm = str(df_avg.loc[idxmax, "hhmm"]) if "hhmm" in df_avg.columns else None
                if price_min is not None and not df_avg.empty:
                    idxmin = df_avg["avg_price"].idxmin()
                    if idxmin is not None and idxmin in df_avg.index:
                        min_hhmm = str(df_avg.loc[idxmin, "hhmm"]) if "hhmm" in df_avg.columns else None

                pct_ann = None
                if price_max is not None and price_min is not None and price_min != 0:
                    pct_ann = ((price_max / price_min) - 1.0) * 250.0 * 100.0
            except Exception:
                price_max = price_min = pct_ann = None
                max_hhmm = min_hhmm = None

            small_css = "font-size:14px; line-height:18px; margin-bottom:4px;"
            val_css = "font-size:18px; font-weight:600;"
            box_css = "padding:8px 10px; border:1px solid #eee; border-radius:6px; background:#fafafa;"

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.markdown(f"<div style='{box_css}'><div style='{small_css}'>Observations totales</div><div style='{val_css}'>" 
                            f"{metadata.get('total_observations', 0):,}" 
                            f"</div></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div style='{box_css}'><div style='{small_css}'>Minutes uniques</div><div style='{val_css}'>" 
                            f"{metadata.get('unique_minutes', 0)}" 
                            f"</div></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div style='{box_css}'><div style='{small_css}'>Timezone</div><div style='{val_css}'>" 
                            f"{metadata.get('timezone', 'UTC')}" 
                            f"</div></div>", unsafe_allow_html=True)
            with col4:
                max_display = (f"{price_max:.3f} @ {max_hhmm}" if price_max is not None and max_hhmm else (f"{price_max:.3f}" if price_max is not None else "N/A"))
                st.markdown(f"<div style='{box_css}'><div style='{small_css}'>Max</div><div style='{val_css}'>" 
                            f"{max_display}" 
                            + "</div></div>", unsafe_allow_html=True)
            with col5:
                min_display = (f"{price_min:.3f} @ {min_hhmm}" if price_min is not None and min_hhmm else (f"{price_min:.3f}" if price_min is not None else "N/A"))
                st.markdown(f"<div style='{box_css}'><div style='{small_css}'>Min</div><div style='{val_css}'>" 
                            f"{min_display}" 
                            + "</div></div>", unsafe_allow_html=True)
            with col6:
                st.markdown(f"<div style='{box_css}'><div style='{small_css}'>Pct annuel estimé</div><div style='{val_css}'>" 
                            f"{pct_ann:.2f}%" if pct_ann is not None else "N/A" 
                            + "</div></div>", unsafe_allow_html=True)

            # Graphique
            st.subheader("Graphique Average Day")
            fig = plot_average_day(df_avg, metadata)
            st.plotly_chart(fig, use_container_width=True)

            # Tableau des données
            st.subheader("Données détaillées")
            st.dataframe(df_avg, use_container_width=True)

            # Téléchargement CSV
            csv = df_avg.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger CSV",
                data=csv,
                file_name=f"{asset}_{sym}_average_day.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Erreur lors du calcul: {e}")

elif page == "Audit Silver":
    st.header("🔎 Audit Silver: Qualité des minutes (UTC)")
    st.write("Analyse des fichiers Silver (parquet) pour un symbole sur tout l'historique: volumes, minutes réelles vs synthétiques, doublons, min/max timestamps, et tableau complet des comptes par minute de la journée (UTC).")

    # Sélection asset/symbole
    def _discover_symbols(asset_type: str) -> list:
        symbols = set()
        base = Path("data") / "silver" / f"asset_class={asset_type}"
        if base.exists():
            for sd in base.iterdir():
                if sd.is_dir() and sd.name.startswith("symbol="):
                    symbols.add(sd.name.replace("symbol=", ""))
        return sorted(symbols)

    asset_type = st.selectbox("Type d'asset", ["stock", "etf", "future", "crypto", "forex", "index"], key="audit_silver_asset")
    symbols = _discover_symbols(asset_type)
    if not symbols:
        st.warning("Aucune donnée Silver pour ce type d'asset.")
        st.stop()

    symbol = st.selectbox("Symbole", symbols, key="audit_silver_symbol")

    # Base du symbole (on prendra toujours tout l'historique)
    base_sym = Path("data") / "silver" / f"asset_class={asset_type}" / f"symbol={symbol}"

    if st.button("Analyser l'historique complet", type="primary"):
        try:
            with st.spinner("Chargement et audit (toutes années)..."):
                metrics, per_minute = audit_silver_tool(Path("data"), asset_type, symbol)

            if metrics.get("rows", 0) == 0:
                st.info("Aucune donnée pour ce symbole.")
                st.stop()

            colA, colB, colC, colD = st.columns(4)
            with colA:
                st.metric("Lignes totales", f"{metrics['rows']:,}")
            with colB:
                st.metric("Timestamps uniques", f"{metrics['unique_ts']:,}")
            with colC:
                st.metric("Doublons", f"{metrics['duplicates']:,}")
            with colD:
                st.metric("Intervalle", f"{metrics['ts_min']} → {metrics['ts_max']}")
            st.caption(f"Timezone utilisée pour l'audit: {metrics.get('timezone_used', 'UTC')}")

            if metrics.get("has_filled_from_ts"):
                synth_total = int(per_minute["synth_count"].sum())
                real_total = int(per_minute["real_count"].sum())
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Minutes réelles", f"{real_total:,}")
                with col2:
                    st.metric("Minutes synthétiques", f"{synth_total:,}")

            st.subheader("Comptes par minute (UTC) — historique complet")
            st.dataframe(per_minute, use_container_width=True)

            # Astuce: pour détail par année ou doublons, utiliser le script CLI audit_silver.py

            st.success("Audit Silver terminé")
        except Exception as e:
            st.error(f"Erreur durant l'audit Silver: {e}")

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
