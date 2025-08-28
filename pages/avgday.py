import streamlit as st
from pathlib import Path
from src.tools.avgday import (
    compute_avgday_by_period, compute_avgday_by_weekday, 
    compute_avgday_all_period, plot_avgday_periods, plot_avgday_weekdays,
    _get_available_periods
)
import pandas as pd


def render():
    """Affiche la page AvgDay (analyses par périodes et jours de semaine)"""
    st.header("📊 AvgDay - Analyses Temporelles")
    st.write("Analyse des profils intraday par périodes et jours de semaine (méthodologie ultra-rapide).")

    def _discover_silver_symbols(asset_type: str) -> list:
        symbols = set()
        base = Path("data") / "silver" / f"asset_class={asset_type}"
        if base.exists():
            for sd in base.iterdir():
                if sd.is_dir() and sd.name.startswith("symbol="):
                    symbols.add(sd.name.replace("symbol=", ""))
        return sorted(symbols)

    def _compute_stats(df):
        """Calcule les stats min/max avec heures et pourcentage annualisé"""
        if df.empty:
            return {}
        
        try:
            price_max = float(df["avg_price"].max())
            price_min = float(df["avg_price"].min())
            
            idxmax = df["avg_price"].idxmax()
            max_hhmm = str(df.loc[idxmax, "hhmm"])
            
            idxmin = df["avg_price"].idxmin()
            min_hhmm = str(df.loc[idxmin, "hhmm"])
            
            pct_ann = ((price_max / price_min) - 1.0) * 250.0 * 100.0 if price_min > 0 else None
            
            return {
                "max_price": price_max,
                "min_price": price_min,
                "max_hhmm": max_hhmm,
                "min_hhmm": min_hhmm,
                "pct_ann": pct_ann,
                "observations": int(df["count"].sum()),
                "unique_minutes": len(df)
            }
        except Exception:
            return {}

    def _display_stats_table(data_dict, title):
        """Affiche un tableau de statistiques avec min/max et pourcentages"""
        if not data_dict:
            return
            
        st.write(f"**{title}:**")
        stats_rows = []
        for name, df in data_dict.items():
            stats = _compute_stats(df)
            if stats:
                stats_rows.append({
                    "Période/Jour": name,
                    "Observations": f"{stats['observations']:,}",
                    "Minutes": stats['unique_minutes'],
                    "Max": f"{stats['max_price']:.3f} @ {stats['max_hhmm']}",
                    "Min": f"{stats['min_price']:.3f} @ {stats['min_hhmm']}",
                    "% Annuel": f"{stats['pct_ann']:.2f}%" if stats['pct_ann'] is not None else "N/A"
                })
        
        if stats_rows:
            stats_df = pd.DataFrame(stats_rows)
            st.dataframe(stats_df, use_container_width=True)

    # Sélection symbole
    asset = st.selectbox("Type d'asset", ["stock", "etf", "future", "crypto", "forex", "index"], key="avgday_asset")
    
    # Interface de recherche pour 5000+ symboles
    search_symbol = st.text_input("🔍 Rechercher symbole", placeholder="Ex: AAPL, MSFT, EUR...", key="avgday_search")
    
    if search_symbol:
        syms = _discover_silver_symbols(asset)
        search_lower = search_symbol.lower()
        filtered_syms = [s for s in syms if search_lower in s.lower()][:50]
        
        if not filtered_syms:
            st.warning(f"Aucun symbole trouvé pour '{search_symbol}' dans {asset}")
            st.stop()
        
        if len(filtered_syms) == 1:
            sym = filtered_syms[0]
            st.success(f"Symbole sélectionné: **{sym}**")
        else:
            st.write(f"**{len(filtered_syms)} résultat(s) trouvé(s):**")
            sym = st.selectbox("Choisir le symbole", filtered_syms, key="avgday_symbol_filtered")
    else:
        syms = _discover_silver_symbols(asset)[:100]
        if not syms:
            st.warning("Aucune donnée Silver disponible pour ce type d'asset.")
            st.stop()
        sym = st.selectbox("Symbole (100 premiers)", syms, key="avgday_symbol")
        if len(_discover_silver_symbols(asset)) > 100:
            st.info("💡 Plus de 100 symboles disponibles. Utilisez la recherche ci-dessus pour plus de choix.")
    
    price_col = st.selectbox("Colonne de prix", ["close", "open", "high", "low"], index=0, key="avgday_price")

    # BOUTON PRINCIPAL - TOUT ANALYSER
    if st.button("📊 Analyser Toutes les Périodes + Weekdays", type="primary", key="calc_all_comprehensive"):
        try:
            # Obtenir les périodes disponibles
            available_periods = _get_available_periods(Path("data"), asset, sym)
            if not available_periods:
                st.warning("Aucune période disponible pour ce symbole.")
                st.stop()

            # Calculer toutes les périodes
            with st.spinner("Calcul de toutes les analyses temporelles..."):
                periods_data = {}
                last_metadata = None
                
                progress = st.progress(0)
                for i, period in enumerate(available_periods):
                    df_result, metadata = compute_avgday_by_period(
                        Path("data"), asset, sym, period, price_col
                    )
                    if not df_result.empty:
                        periods_data[period] = df_result
                        last_metadata = metadata
                    progress.progress((i + 1) / (len(available_periods) + 1))
                
                # Calculer weekdays
                weekdays_data, weekday_metadata = compute_avgday_by_weekday(
                    Path("data"), asset, sym, price_col
                )
                progress.progress(1.0)

            if not periods_data and not weekdays_data:
                st.error("Aucune donnée trouvée pour ce symbole.")
                st.stop()

            st.success(f"✅ Analyses complètes: {len(periods_data)} périodes + {len(weekdays_data)} jours de semaine")

            # 1. GRAPHIQUE GÉNÉRAL (average day sur TOUTE la période dans la base)
            st.subheader("1️⃣ Average Day Global - Toute la Période en Base")
            
            # Calculer l'average day sur TOUTE la période disponible
            try:
                with st.spinner("Calcul average day global (toute la période en base)..."):
                    df_global, metadata_global = compute_avgday_all_period(Path("data"), asset, sym, price_col)
                
                if not df_global.empty:
                    global_data = {"Toute la période": df_global}
                    _display_stats_table(global_data, "Statistiques Période Globale")
                    
                    # Afficher les bornes temporelles
                    start_date = metadata_global.get('start_date', 'Inconnue')[:10]
                    end_date = metadata_global.get('end_date', 'Inconnue')[:10]
                    st.info(f"📅 Période complète: **{start_date}** → **{end_date}**")
                    
                    fig_global = plot_avgday_periods(global_data, metadata_global, f"Average Day Global - {start_date} à {end_date}")
                    st.plotly_chart(fig_global, use_container_width=True)
                else:
                    st.error(f"Impossible de calculer la période globale: {metadata_global.get('error', 'Erreur inconnue')}")
            except Exception as e:
                st.error(f"Erreur calcul global: {e}")

            # 2. GRAPHIQUE COURT TERME (1 mois, 6 mois, 3 ans)
            short_term_periods = {}
            for period in ["dernier_mois", "6_derniers_mois", "3_dernieres_annees"]:
                if period in periods_data:
                    short_term_periods[period] = periods_data[period]
            
            if short_term_periods:
                st.subheader("2️⃣ Périodes Récentes (1 mois / 6 mois / 3 ans)")
                # Renommage pour affichage plus lisible
                display_short = {}
                name_mapping = {
                    "dernier_mois": "1 Mois",
                    "6_derniers_mois": "6 Mois", 
                    "3_dernieres_annees": "3 Ans"
                }
                for k, v in short_term_periods.items():
                    display_short[name_mapping.get(k, k)] = v
                
                _display_stats_table(display_short, "Statistiques Périodes Récentes")
                fig_recent = plot_avgday_periods(display_short, last_metadata or {}, "Périodes Récentes")
                st.plotly_chart(fig_recent, use_container_width=True)

            # 3. GRAPHIQUES HISTORIQUES (UN GRAPHIQUE INDIVIDUEL PAR PÉRIODE DE 4 ANS)
            historical_periods = {k: v for k, v in periods_data.items() if k.startswith("4ans_")}
            
            if historical_periods:
                st.subheader("3️⃣ Périodes Historiques (un graphique individuel par période de 4 ans)")
                
                # Trier les périodes de 4 ans par année décroissante
                sorted_historical = sorted(historical_periods.items(), 
                                         key=lambda x: int(x[0].split("_")[1]), 
                                         reverse=True)

                # UN GRAPHIQUE INDIVIDUEL pour chaque période de 4 ans
                for i, (period_key, df_period) in enumerate(sorted_historical):
                    try:
                        year_end = int(period_key.split("_")[1])
                        year_start = year_end - 3  # 4 ans: année fin - 3
                        
                        # Titre et stats pour cette période de 4 ans
                        st.write(f"### 📊 Période Historique {i+1}: {year_start}-{year_end}")
                        
                        # Données pour ce graphique individuel (une seule période)
                        individual_data = {f"{year_start}-{year_end}": df_period}
                        _display_stats_table(individual_data, f"Stats {year_start}-{year_end}")
                        
                        # GRAPHIQUE INDIVIDUEL pour cette période
                        title = f"Average Day {year_start}-{year_end}"
                        fig_hist = plot_avgday_periods(individual_data, last_metadata or {}, title)
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Espacement entre les graphiques
                        st.markdown("---")
                        
                    except Exception as e:
                        st.warning(f"Erreur période {period_key}: {e}")

            # 4. GRAPHIQUE WEEKDAYS
            if weekdays_data:
                st.subheader("4️⃣ Analyse par Jour de Semaine (3 dernières années)")
                
                total_obs = weekday_metadata.get('total_observations', 0)
                st.info(f"📊 Total observations sur 3 ans: **{total_obs:,}** | Timezone: **{weekday_metadata.get('timezone', 'UTC')}**")
                
                _display_stats_table(weekdays_data, "Statistiques par Jour de Semaine")
                
                fig_weekdays = plot_avgday_weekdays(weekdays_data, weekday_metadata)
                st.plotly_chart(fig_weekdays, use_container_width=True)

            # 5. TÉLÉCHARGEMENTS
            with st.expander("📥 Téléchargements"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Périodes:**")
                    for period, df in periods_data.items():
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            f"CSV {period}",
                            data=csv,
                            file_name=f"{asset}_{sym}_avgday_{period}.csv",
                            mime="text/csv",
                            key=f"dl_period_{period}"
                        )
                
                with col2:
                    st.write("**Jours de semaine:**")
                    for weekday, df in weekdays_data.items():
                        if not df.empty:
                            csv = df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                f"CSV {weekday}",
                                data=csv,
                                file_name=f"{asset}_{sym}_avgday_{weekday.lower()}.csv",
                                mime="text/csv",
                                key=f"dl_weekday_{weekday}"
                            )

        except Exception as e:
            st.error(f"Erreur lors du calcul: {e}")
            st.exception(e)

    # Section pour tester une période spécifique
    with st.expander("🔧 Test Période Spécifique"):
        try:
            available_periods = _get_available_periods(Path("data"), asset, sym)
            if available_periods:
                selected_period = st.selectbox("Période de test", available_periods, key="test_period")
                if st.button("Tester", key="test_calc"):
                    df_result, metadata = compute_avgday_by_period(Path("data"), asset, sym, selected_period, price_col)
                    if not df_result.empty:
                        st.success(f"✅ Période {selected_period}: {len(df_result)} minutes")
                        stats = _compute_stats(df_result)
                        if stats:
                            st.write(f"**Stats:** {stats['observations']:,} obs, Max: {stats['max_price']:.3f} @ {stats['max_hhmm']}, Min: {stats['min_price']:.3f} @ {stats['min_hhmm']}, % Ann: {stats['pct_ann']:.2f}%")
                        st.dataframe(df_result.head(10))
                    else:
                        st.error(f"❌ Aucune donnée pour {selected_period}: {metadata.get('error', 'Erreur inconnue')}")
        except Exception as e:
            st.error(f"Erreur test: {e}")