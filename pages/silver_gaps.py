import streamlit as st
import pandas as pd
import pyarrow.dataset as ds
from pathlib import Path
from src.calendar.session_loader import TradingSessionTemplates, SymbolSessionRegistry
from src.silver.run_fill import run_gap_fill_for_symbol


def render():
    """Affiche la page Silver - Fill Gaps"""
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
