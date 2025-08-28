import streamlit as st
from pathlib import Path
from src.tools.average_day import compute_average_day, plot_average_day


def render():
    """Affiche la page Average Day"""
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
