import streamlit as st
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

from src.tools.avgday_pct import compute_average_day_pct, compute_average_day_pct_period, plot_average_day_pct


def render():
    st.header("📊 AvgDay % - Performance intraday moyenne")
    st.write("Calcul en pourcentage par rapport à la première minute de chaque session (0% à l'ouverture).")

    def _discover_silver_symbols(asset_type: str) -> list:
        symbols = set()
        base = Path("data") / "silver" / f"asset_class={asset_type}"
        if base.exists():
            for sd in base.iterdir():
                if sd.is_dir() and sd.name.startswith("symbol="):
                    symbols.add(sd.name.replace("symbol=", ""))
        return sorted(symbols)

    asset = st.selectbox("Type d'asset", ["stock", "etf", "future", "crypto", "forex", "index"], key="avgdaypct_asset")
    search_symbol = st.text_input("🔍 Rechercher symbole", placeholder="Ex: AAPL, ZW, EUR...", key="avgdaypct_search")

    if search_symbol:
        syms = _discover_silver_symbols(asset)
        search_lower = search_symbol.lower()
        filtered = [s for s in syms if search_lower in s.lower()][:50]
        if not filtered:
            st.warning(f"Aucun symbole trouvé pour '{search_symbol}' dans {asset}")
            st.stop()
        symbol = st.selectbox("Choisir le symbole", filtered, key="avgdaypct_symbol_filtered")
    else:
        syms = _discover_silver_symbols(asset)[:100]
        if not syms:
            st.warning("Aucune donnée Silver disponible pour ce type d'asset.")
            st.stop()
        symbol = st.selectbox("Symbole (100 premiers)", syms, key="avgdaypct_symbol")
        if len(_discover_silver_symbols(asset)) > 100:
            st.info("💡 Plus de 100 symboles disponibles. Utilisez la recherche ci-dessus pour plus de choix.")

    price_col = st.selectbox("Colonne de prix", ["close", "open", "high", "low"], index=0, key="avgdaypct_price")

    def _display_stats_and_chart(df, metadata, title, chart_key):
        """Affiche les stats compactes + graphique pour une période."""
        if df.empty:
            st.error(f"Erreur: {metadata.get('error', 'Aucune donnée')}")
            return

        # Stats compactes avec performance annualisée
        try:
            pct_max = float(df["avg_pct"].max())
            pct_min = float(df["avg_pct"].min()) 
            max_hhmm = df.loc[df["avg_pct"].idxmax(), "hhmm"]
            min_hhmm = df.loc[df["avg_pct"].idxmin(), "hhmm"]
            # Performance annualisée : (max - min) * 250 jours de bourse
            perf_annualisee = (pct_max - pct_min) * 250.0
        except Exception:
            pct_max = pct_min = max_hhmm = min_hhmm = perf_annualisee = None

        small_css = "font-size:14px; line-height:18px; margin-bottom:4px;"
        val_css = "font-size:18px; font-weight:600;"
        box_css = "padding:8px 10px; border:1px solid #eee; border-radius:6px; background:#fafafa;"

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"<div style='{box_css}'><div style='{small_css}'>Observations</div><div style='{val_css}'>" 
                        f"{metadata.get('total_observations', 0):,}" 
                        f"</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div style='{box_css}'><div style='{small_css}'>Sessions</div><div style='{val_css}'>" 
                        f"{metadata.get('total_sessions', 0):,}" 
                        f"</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div style='{box_css}'><div style='{small_css}'>Max %</div><div style='{val_css}'>" 
                        f"{pct_max:.3f}% @ {max_hhmm}" if pct_max is not None else "N/A" 
                        + "</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div style='{box_css}'><div style='{small_css}'>Min %</div><div style='{val_css}'>" 
                        f"{pct_min:.3f}% @ {min_hhmm}" if pct_min is not None else "N/A" 
                        + "</div></div>", unsafe_allow_html=True)
        with c5:
            st.markdown(f"<div style='{box_css}'><div style='{small_css}'>Perf Annualisée</div><div style='{val_css}'>" 
                        f"{perf_annualisee:.1f}%" if perf_annualisee is not None else "N/A" 
                        + "</div></div>", unsafe_allow_html=True)

        # Graphique
        fig = plot_average_day_pct(df, metadata, title)
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

        return df

    if st.button("📊 Calculer AvgDay % (Toutes Périodes)", type="primary", key="avgdaypct_calc"):
        current_date = datetime.now()
        
        # 1. GRAPHIQUE GLOBAL (toute la période)
        st.subheader("1️⃣ AvgDay % Global - Toute la Période")
        with st.spinner("Calcul global..."):
            df_global, metadata_global = compute_average_day_pct(Path("data"), asset, symbol, price_col)
        
        df_result_global = _display_stats_and_chart(
            df_global, metadata_global, 
            f"AvgDay % Global - {symbol}", 
            "chart_global"
        )

        # 2. GRAPHIQUE DERNIÈRE ANNÉE
        st.subheader("2️⃣ AvgDay % Dernière Année")
        start_last_year = current_date - timedelta(days=365)
        with st.spinner("Calcul dernière année..."):
            df_last_year, metadata_last_year = compute_average_day_pct_period(
                Path("data"), asset, symbol, start_last_year, current_date, price_col
            )
        
        df_result_last_year = _display_stats_and_chart(
            df_last_year, metadata_last_year,
            f"AvgDay % Dernière Année - {symbol}",
            "chart_last_year"
        )

        # 3. GRAPHIQUE 4 ANNÉES D'AVANT (il y a 1 an → il y a 5 ans)
        st.subheader("3️⃣ AvgDay % Période Historique (4 années d'avant)")
        start_historical = current_date - timedelta(days=5*365)  # Il y a 5 ans
        end_historical = current_date - timedelta(days=365)      # Il y a 1 an
        with st.spinner("Calcul période historique..."):
            df_historical, metadata_historical = compute_average_day_pct_period(
                Path("data"), asset, symbol, start_historical, end_historical, price_col
            )
        
        df_result_historical = _display_stats_and_chart(
            df_historical, metadata_historical,
            f"AvgDay % Historique (4 ans avant) - {symbol}",
            "chart_historical"
        )

        # 4. TÉLÉCHARGEMENTS
        st.subheader("📥 Téléchargements")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if df_result_global is not None and not df_result_global.empty:
                csv_global = df_result_global.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "CSV Global",
                    data=csv_global,
                    file_name=f"{asset}_{symbol}_avgday_pct_global.csv",
                    mime="text/csv",
                    key="dl_global"
                )
        
        with col2:
            if df_result_last_year is not None and not df_result_last_year.empty:
                csv_last_year = df_result_last_year.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "CSV Dernière Année",
                    data=csv_last_year,
                    file_name=f"{asset}_{symbol}_avgday_pct_last_year.csv",
                    mime="text/csv",
                    key="dl_last_year"
                )
        
        with col3:
            if df_result_historical is not None and not df_result_historical.empty:
                csv_historical = df_result_historical.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "CSV Historique",
                    data=csv_historical,
                    file_name=f"{asset}_{symbol}_avgday_pct_historical.csv",
                    mime="text/csv",
                    key="dl_historical"
                )
