import streamlit as st
import pandas as pd
from pathlib import Path
from src.tools.fetch_prices import fetch_prices, fetch_prices_at_hhmm


def render():
    """Affiche la page Récup Prix"""
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
