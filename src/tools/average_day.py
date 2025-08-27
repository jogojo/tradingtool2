import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import plotly.graph_objects as go
import plotly.express as px

try:
    # Mapping sessions → timezone si disponible
    from src.calendar.session_loader import TradingSessionTemplates, SymbolSessionRegistry
except Exception:
    TradingSessionTemplates = None
    SymbolSessionRegistry = None


def _resolve_session_timezone(symbol: str) -> Tuple[str, Optional[Dict]]:
    """Résoud la timezone et session d'un symbole via le registry."""
    if TradingSessionTemplates is None or SymbolSessionRegistry is None:
        return "UTC", None
    try:
        templates = TradingSessionTemplates()
        reg = SymbolSessionRegistry()
        tpl_name = reg.get(symbol)
        if tpl_name and tpl_name in templates.templates:
            session_cfg = templates.templates[tpl_name]
            return session_cfg.get("timezone", "UTC"), session_cfg
    except Exception:
        pass
    return "UTC", None


def compute_average_day(base_dir: Path, asset_class: str, symbol: str, price_col: str = "close") -> Tuple[pd.DataFrame, Dict]:
    """
    Calcule l'average day VRAIMENT optimisé:
    - Groupement sur minute_du_jour (0-1439) pas sur strings HH:MM
    - np.bincount vectorisé pour SUM/COUNT
    - Conversion timezone une seule fois par batch
    - Zéro boucle Python sur les groupes
    """
    import numpy as np
    
    base_sym = base_dir / "silver" / f"asset_class={asset_class}" / f"symbol={symbol}"
    if not base_sym.exists():
        return pd.DataFrame(columns=["hhmm", "avg_price", "count"]), {"error": "Symbole non trouvé"}

    # Résolution timezone + session
    local_tz, session_cfg = _resolve_session_timezone(symbol)
    
    # Dataset ultra-minimal: timestamp + price seulement
    dset = ds.dataset(base_sym, format="parquet")
    required_cols = ["timestamp", price_col]
    cols = [c for c in required_cols if c in dset.schema.names]
    if price_col not in cols:
        return pd.DataFrame(columns=["hhmm", "avg_price", "count"]), {"error": f"Colonne {price_col} manquante"}

    scanner = dset.scanner(columns=cols)
    
    # Extraction des heures de session (en minutes depuis minuit)
    session_start_min = None
    session_end_min = None
    if session_cfg and "session" in session_cfg:
        try:
            from datetime import time
            sess = session_cfg["session"]
            start_time = time.fromisoformat(sess["open"])
            end_time = time.fromisoformat(sess["close"])
            session_start_min = start_time.hour * 60 + start_time.minute
            session_end_min = end_time.hour * 60 + end_time.minute
        except Exception:
            pass

    # Accumulateurs vectorisés (1440 minutes par jour)
    price_sums = np.zeros(1440, dtype=np.float64)
    price_counts = np.zeros(1440, dtype=np.int64)
    total_rows = 0
    
    # Stream optimisé par batches
    for batch_group in scanner.scan_batches():
        try:
            batch = batch_group.to_record_batch()
        except Exception:
            batch = getattr(batch_group, "record_batch", batch_group)
        
        if batch.num_rows == 0:
            continue
            
        ts_col = batch.column(0)  # timestamp
        price_col_idx = batch.column(1)  # prix
        
        try:
            # Conversion timezone UNE SEULE FOIS par batch
            if local_tz == "UTC":
                ts_local = ts_col
            else:
                ts_local = pc.assume_timezone(ts_col, timezone="UTC")
                ts_local = pc.cast(ts_local, pa.timestamp("ns", tz=local_tz))
            
            # Extraction heure/minute en entiers (pas strings!)
            hour = pc.hour(ts_local)
            minute = pc.minute(ts_local)
            
            # CLEF: minute_du_jour = hour * 60 + minute (0-1439)
            minute_of_day = pc.add(pc.multiply(hour, 60), minute)
            
            # Filtre session si défini (en entiers, pas strings)
            if session_start_min is not None and session_end_min is not None:
                if session_start_min <= session_end_min:
                    # Session normale (ex: 570-960 pour 09:30-16:00)
                    session_mask = pc.and_(
                        pc.greater_equal(minute_of_day, session_start_min),
                        pc.less(minute_of_day, session_end_min)
                    )
                else:
                    # Session overnight (ex: 1080-570 pour 18:00-09:30)
                    session_mask = pc.or_(
                        pc.greater_equal(minute_of_day, session_start_min),
                        pc.less(minute_of_day, session_end_min)
                    )
                
                # Appliquer le filtre
                minute_of_day = pc.filter(minute_of_day, session_mask)
                price_col_idx = pc.filter(price_col_idx, session_mask)
            
            # Conversion en numpy pour bincount (ultra-rapide)
            minute_idx = minute_of_day.to_numpy()
            prices = price_col_idx.to_numpy()
            
            # Enlever NaN
            valid_mask = ~(np.isnan(prices) | np.isnan(minute_idx.astype(float)))
            minute_idx = minute_idx[valid_mask].astype(np.int32)
            prices = prices[valid_mask]
            
            # Filtre minute_idx valides (0-1439)
            valid_minutes = (minute_idx >= 0) & (minute_idx < 1440)
            minute_idx = minute_idx[valid_minutes]
            prices = prices[valid_minutes]
            
            if len(minute_idx) > 0:
                # VECTORISATION PURE: bincount pour SUM et COUNT
                batch_sums = np.bincount(minute_idx, weights=prices, minlength=1440)
                batch_counts = np.bincount(minute_idx, minlength=1440)
                
                # Accumulation vectorisée (pas de boucles Python!)
                price_sums += batch_sums
                price_counts += batch_counts
                total_rows += len(minute_idx)
            
        except Exception as e:
            # Fallback numpy si Arrow échoue
            df_batch = batch.to_pandas()
            df_batch = df_batch.dropna()
            if df_batch.empty:
                continue
                
            df_batch["timestamp"] = pd.to_datetime(df_batch["timestamp"], utc=True)
            
            # Conversion timezone
            if local_tz == "UTC":
                ts_local = df_batch["timestamp"]
            else:
                try:
                    ts_local = df_batch["timestamp"].dt.tz_convert(local_tz)
                except Exception:
                    ts_local = df_batch["timestamp"]
            
            # Minute du jour (entier)
            minute_of_day = ts_local.dt.hour * 60 + ts_local.dt.minute
            
            # Filtre session
            if session_start_min is not None and session_end_min is not None:
                if session_start_min <= session_end_min:
                    mask = (minute_of_day >= session_start_min) & (minute_of_day < session_end_min)
                else:
                    mask = (minute_of_day >= session_start_min) | (minute_of_day < session_end_min)
                
                minute_of_day = minute_of_day[mask]
                prices = df_batch[price_col].iloc[mask.values]
            else:
                prices = df_batch[price_col]
            
            # Vectorisation numpy
            minute_idx = minute_of_day.values.astype(np.int32)
            prices_vals = prices.values
            
            # Filtre valides
            valid_mask = ~(np.isnan(prices_vals) | (minute_idx < 0) | (minute_idx >= 1440))
            minute_idx = minute_idx[valid_mask]
            prices_vals = prices_vals[valid_mask]
            
            if len(minute_idx) > 0:
                batch_sums = np.bincount(minute_idx, weights=prices_vals, minlength=1440)
                batch_counts = np.bincount(minute_idx, minlength=1440)
                
                price_sums += batch_sums
                price_counts += batch_counts
                total_rows += len(minute_idx)

    # Calcul des moyennes vectorisé
    valid_minutes = price_counts > 0
    avg_prices = np.divide(price_sums, price_counts, out=np.zeros_like(price_sums), where=valid_minutes)
    
    # Conversion finale vers HH:MM (seulement pour l'affichage)
    result_data = []
    for minute_idx in np.where(valid_minutes)[0]:
        hour = minute_idx // 60
        minute = minute_idx % 60
        hhmm = f"{hour:02d}:{minute:02d}"
        result_data.append({
            "hhmm": hhmm,
            "avg_price": float(avg_prices[minute_idx]),
            "count": int(price_counts[minute_idx])
        })
    
    df_result = pd.DataFrame(result_data)
    
    metadata = {
        "symbol": symbol,
        "asset_class": asset_class,
        "price_col": price_col,
        "timezone": local_tz,
        "session_cfg": session_cfg,
        "total_observations": int(total_rows),
        "unique_minutes": int(valid_minutes.sum())
    }
    
    return df_result, metadata


def plot_average_day(df: pd.DataFrame, metadata: Dict, title: Optional[str] = None) -> go.Figure:
    """Trace le graphique de l'average day avec Plotly."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Aucune donnée", x=0.5, y=0.5, showarrow=False)
        return fig
    
    if title is None:
        symbol = metadata.get("symbol", "Unknown")
        price_col = metadata.get("price_col", "price")
        tz = metadata.get("timezone", "UTC")
        title = f"Average Day - {symbol} ({price_col}) - TZ: {tz}"
    
    fig = go.Figure()
    
    # Ligne principale
    fig.add_trace(go.Scatter(
        x=df["hhmm"],
        y=df["avg_price"],
        mode='lines+markers',
        name=f'Avg {metadata.get("price_col", "price")}',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    # Barres de volume (count) en arrière-plan
    fig.add_trace(go.Bar(
        x=df["hhmm"],
        y=df["count"],
        name='Observations',
        yaxis='y2',
        opacity=0.3,
        marker_color='lightgray'
    ))
    
    # Layout
    fig.update_layout(
        title=title,
        xaxis_title="Heure (TZ locale)",
        yaxis_title=f"Prix moyen ({metadata.get('price_col', 'price')})",
        yaxis2=dict(
            title="Nombre d'observations",
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=600,
        showlegend=True
    )
    
    # Rotation des labels X si trop nombreux
    if len(df) > 20:
        fig.update_xaxes(tickangle=45)
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Calcul Average Day ultra-rapide")
    parser.add_argument("--base", type=str, default="./data")
    parser.add_argument("--asset", type=str, required=True)
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--price", type=str, default="close", help="Colonne de prix (close, open, high, low)")
    parser.add_argument("--plot", action="store_true", help="Afficher le graphique")
    args = parser.parse_args()

    df, metadata = compute_average_day(Path(args.base), args.asset, args.symbol, args.price)
    
    print("== Metadata ==")
    for k, v in metadata.items():
        if k != "session_cfg":
            print(f"{k}: {v}")
    
    print("\n== Average Day ==")
    print(df.to_string(index=False))
    
    if args.plot and not df.empty:
        fig = plot_average_day(df, metadata)
        fig.show()


if __name__ == "__main__":
    main()
