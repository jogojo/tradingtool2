import streamlit as st
from pathlib import Path
from src.tools.audit_silver import audit_silver as audit_silver_tool


def render():
    """Affiche la page Audit Silver"""
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
