import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


def fetch_prices(base_dir: Path, asset_class: str, symbol: str, timestamps_utc: Iterable[pd.Timestamp]) -> pd.DataFrame:
    """
    Récupère OHLCV (+ filled_from_ts si présent) pour une liste de timestamps UTC (ns).
    OPTIMISÉ: lit uniquement les années nécessaires.
    """
    base_sym = base_dir / "silver" / f"asset_class={asset_class}" / f"symbol={symbol}"
    if not base_sym.exists():
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","filled_from_ts"])

    ts_parsed = pd.to_datetime(list(timestamps_utc), utc=True, errors="raise")
    if len(ts_parsed) == 0:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","filled_from_ts"])
    
    # OPTIMISATION: Lire uniquement les années nécessaires
    years_needed = sorted(ts_parsed.dt.year.unique())
    targets_set = set(ts_parsed.astype("datetime64[ns, UTC]"))

    parts: List[pd.DataFrame] = []
    for year in years_needed:
        year_dir = base_sym / f"year={year}"
        if not year_dir.exists():
            continue
        # Lire uniquement cette année
        dset = ds.dataset(year_dir, format="parquet")
        cols = [c for c in ["timestamp","open","high","low","close","volume","filled_from_ts"] if c in dset.schema.names]
        tbl = dset.to_table(columns=cols)
        df = tbl.to_pandas()
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # Filtre exact en mémoire (rapide car une seule année)
        mask = df["timestamp"].isin(targets_set)
        if mask.any():
            parts.append(df.loc[mask])

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["timestamp","open","high","low","close","volume","filled_from_ts"])
    return out.sort_values("timestamp") if not out.empty else out


def fetch_prices_at_hhmm(base_dir: Path, asset_class: str, symbol: str, hhmm_utc: str, timezone: str = "UTC") -> pd.DataFrame:
    """
    Récupère toutes les lignes Silver dont l'heure-minutes locale (timezone) == hh:mm fourni.
    ULTRA-OPTIMISÉ: utilise Arrow compute natif avec filtre HH:MM direct.
    """
    base_sym = base_dir / "silver" / f"asset_class={asset_class}" / f"symbol={symbol}"
    if not base_sym.exists():
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","filled_from_ts"])

    # Dataset global avec filtre Arrow natif (plus rapide que boucle année par année)
    dset = ds.dataset(base_sym, format="parquet")
    cols = [c for c in ["timestamp","open","high","low","close","volume","filled_from_ts"] if c in dset.schema.names]
    
    # Optimisation: utilise Arrow compute pour filtrer directement en C++
    import pyarrow.compute as pc
    
    # STRATÉGIE: Créer un filtre Arrow sur l'heure/minute
    try:
        # Parse hh:mm
        hh, mm = map(int, hhmm_utc.split(":"))
        
        # Scan avec projection des colonnes utiles seulement
        scanner = dset.scanner(columns=cols)
        
        # Filtrage en streaming par batches avec Arrow compute (C++)
        matched_batches = []
        for batch_group in scanner.scan_batches():
            try:
                batch = batch_group.to_record_batch()
            except Exception:
                batch = getattr(batch_group, "record_batch", batch_group)
            
            ts_col = batch.column(batch.schema.get_field_index("timestamp"))
            
            # Conversion timezone en Arrow (si supporté)
            try:
                if timezone != "UTC":
                    ts_local = pc.assume_timezone(ts_col, timezone="UTC")
                    ts_local = pc.cast(ts_local, pa.timestamp("ns", tz=timezone))
                else:
                    ts_local = ts_col
                
                # Extraction heure/minute en Arrow
                hour = pc.hour(ts_local)
                minute = pc.minute(ts_local)
                
                # Filtre combiné: heure == hh ET minute == mm
                mask = pc.and_(pc.equal(hour, hh), pc.equal(minute, mm))
                
                # Appliquer le filtre si des matches
                if pc.any(mask).as_py():
                    filtered_batch = {}
                    for col_name in cols:
                        col_idx = batch.schema.get_field_index(col_name)
                        filtered_batch[col_name] = pc.filter(batch.column(col_idx), mask)
                    matched_batches.append(pa.Table.from_arrays(
                        [filtered_batch[name] for name in cols], names=cols
                    ))
                        
            except Exception:
                # Fallback pandas si Arrow timezone non supporté
                df_batch = batch.to_pandas()
                if df_batch.empty:
                    continue
                df_batch["timestamp"] = pd.to_datetime(df_batch["timestamp"], utc=True)
                try:
                    df_local = df_batch["timestamp"].dt.tz_convert(timezone)
                except Exception:
                    df_local = df_batch["timestamp"]  # garde UTC si conversion échoue
                
                hhmm_mask = (df_local.dt.hour == hh) & (df_local.dt.minute == mm)
                if hhmm_mask.any():
                    matched_batches.append(pa.Table.from_pandas(df_batch.loc[hhmm_mask, cols]))

        # Concaténer tous les batches matchés
        if matched_batches:
            result_table = pa.concat_tables(matched_batches)
            out = result_table.to_pandas()
            out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
            return out.sort_values("timestamp")
        else:
            return pd.DataFrame(columns=cols)
            
    except Exception as e:
        # Fallback complet si Arrow compute échoue
        parts: List[pd.DataFrame] = []
        for year_dir in sorted(base_sym.iterdir()):
            if not (year_dir.is_dir() and year_dir.name.startswith("year=")):
                continue
            dset_year = ds.dataset(year_dir, format="parquet")
            tbl = dset_year.to_table(columns=cols)
            df = tbl.to_pandas()
            if df.empty:
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            try:
                df_local = df["timestamp"].dt.tz_convert(timezone)
            except Exception:
                df_local = df["timestamp"]
            hhmm_mask = df_local.dt.strftime("%H:%M") == hhmm_utc
            if hhmm_mask.any():
                parts.append(df.loc[hhmm_mask])
        
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
        return out.sort_values("timestamp") if not out.empty else out


def main():
    parser = argparse.ArgumentParser(description="Fetch prices from Silver at exact UTC minutes")
    parser.add_argument("--base", type=str, default="./data")
    parser.add_argument("--asset", type=str, required=True)
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--ts", type=str, nargs='+', help="List of ISO UTC timestamps (e.g., 2024-03-05T23:03:00Z)")
    parser.add_argument("--hhmm", type=str, help="UTC hh:mm filter (e.g., 23:03)")
    args = parser.parse_args()

    if args.hhmm:
        df = fetch_prices_at_hhmm(Path(args.base), args.asset, args.symbol, args.hhmm)
    else:
        ts_parsed = [pd.to_datetime(t, utc=True, errors="raise") for t in (args.ts or [])]
        df = fetch_prices(Path(args.base), args.asset, args.symbol, ts_parsed)
    print(df.to_csv(index=False))


if __name__ == "__main__":
    main()


