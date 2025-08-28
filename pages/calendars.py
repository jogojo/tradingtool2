import streamlit as st
from pathlib import Path
from src.calendar.session_loader import TradingSessionTemplates, SymbolSessionRegistry


def render():
    """Affiche la page Calendriers"""
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
                                        if symbol not in reg.map:
                                            reg.set(symbol, template_to_apply)
                                            mapped_count += 1
                                            details.append(f"{symbol} → {template_to_apply}")
                    
                    # Cas 2: Structure plate (fichiers CSV/TXT/ZIP)
                    if not has_organized_structure:
                        for file_path in base_path.rglob("*"):
                            if file_path.is_file() and file_path.suffix.lower() in [".csv", ".txt", ".zip"]:
                                symbol = extract_symbol_from_filename(file_path.name)
                                if symbol:
                                    total_count += 1
                                    if symbol not in reg.map:
                                        reg.set(symbol, template_to_apply)
                                        mapped_count += 1
                                        details.append(f"{symbol} → {template_to_apply}")
                
                st.success(f"✅ Auto-mapping terminé: {mapped_count} nouveaux mappings sur {total_count} symboles trouvés")
                
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
            if stats["templates_used"]:
                most_used = max(stats["templates_used"].items(), key=lambda x: x[1])
                st.write(f"**Template principal:** {most_used[0]} ({most_used[1]} symboles)")
        
        search_term = st.text_input("🔍 Rechercher un symbole", key="search_individual")
        
        if search_term:
            found_symbols = reg.search_symbols(search_term)
            
            if found_symbols:
                symbol_to_edit = st.selectbox("Symbole à configurer", found_symbols, key="symbol_edit")
                current_template = reg.get(symbol_to_edit)
                
                if current_template:
                    st.info(f"Mapping actuel: **{symbol_to_edit}** → `{current_template}`")
                else:
                    st.warning(f"Aucun mapping pour **{symbol_to_edit}**")
                
                new_template = st.selectbox(
                    "Nouveau template",
                    list(templates.templates.keys()),
                    index=list(templates.templates.keys()).index(current_template) if current_template in templates.templates else 0,
                    key="new_template"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Mettre à jour", type="primary"):
                        reg.set(symbol_to_edit, new_template)
                        st.success(f"✅ {symbol_to_edit} → {new_template}")
                        st.rerun()
                
                with col2:
                    if current_template and st.button("🗑️ Supprimer mapping"):
                        reg.remove(symbol_to_edit)
                        st.success(f"🗑️ Mapping supprimé pour {symbol_to_edit}")
                        st.rerun()
            else:
                st.info(f"Aucun symbole trouvé pour '{search_term}'")
