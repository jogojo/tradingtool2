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
    """Résout la timezone et la session d'un symbole via le registry.

    Exigence: pas de fallback UTC. Si aucune règle n'existe, on lève une erreur
    explicite afin d'éviter toute interprétation erronée.
    """
    if TradingSessionTemplates is None or SymbolSessionRegistry is None:
        raise ValueError("Configuration des sessions indisponible. Veuillez vérifier l'installation et les imports.")

    templates = TradingSessionTemplates()
    reg = SymbolSessionRegistry()
    tpl_name = reg.get(symbol)

    if not tpl_name:
        raise ValueError(f"Aucune règle de session trouvée pour '{symbol}'. Mappez le symbole dans la page Calendriers.")
    if tpl_name not in templates.templates:
        raise ValueError(f"Template de session '{tpl_name}' introuvable pour '{symbol}'. Vérifiez config/trading_sessions.json.")

    session_cfg = templates.templates[tpl_name]
    return session_cfg.get("timezone", "UTC"), session_cfg


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

    # Résolution timezone + session (sans fallback; retourne erreur explicite)
    try:
        local_tz, session_cfg = _resolve_session_timezone(symbol)
    except Exception as e:
        return pd.DataFrame(columns=["hhmm", "avg_price", "count"]), {"error": str(e)}
    
    # Dataset ultra-minimal: timestamp + price (+ volume si présent)
    dset = ds.dataset(base_sym, format="parquet")
    required_cols = ["timestamp", price_col]
    if "volume" in dset.schema.names:
        required_cols.append("volume")
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
    has_volume = "volume" in cols
    if has_volume:
        volume_sums = np.zeros(1440, dtype=np.float64)
        volume_counts = np.zeros(1440, dtype=np.int64)
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
        vol_col_idx = None
        if has_volume:
            # index de la colonne volume dans le batch courant
            vol_idx = batch.schema.get_field_index("volume")
            if vol_idx != -1:
                vol_col_idx = batch.column(vol_idx)
        
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
                if vol_col_idx is not None:
                    vol_col_idx = pc.filter(vol_col_idx, session_mask)
            
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

            # Volume moyen (si présent)
            if has_volume and vol_col_idx is not None:
                vols = vol_col_idx.to_numpy()
                vmask = ~(np.isnan(vols) | np.isnan(minute_idx.astype(float)))
                v_idx = minute_idx[vmask]
                v_vals = vols[vmask]
                if len(v_idx) > 0:
                    v_sums = np.bincount(v_idx, weights=v_vals, minlength=1440)
                    v_counts = np.bincount(v_idx, minlength=1440)
                    volume_sums += v_sums
                    volume_counts += v_counts
            
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
                vols_pd = df_batch["volume"].iloc[mask.values] if has_volume and "volume" in df_batch.columns else None
            else:
                prices = df_batch[price_col]
                vols_pd = df_batch["volume"] if has_volume and "volume" in df_batch.columns else None
            
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

            # Volume
            if has_volume and vols_pd is not None:
                vols_vals = pd.to_numeric(vols_pd, errors="coerce").values
                v_valid = ~(np.isnan(vols_vals) | (minute_idx < 0) | (minute_idx >= 1440))
                v_idx = minute_idx[v_valid]
                v_vals = vols_vals[v_valid]
                if len(v_idx) > 0:
                    v_sums = np.bincount(v_idx, weights=v_vals, minlength=1440)
                    v_counts = np.bincount(v_idx, minlength=1440)
                    volume_sums += v_sums
                    volume_counts += v_counts

    # Calcul des moyennes vectorisé
    valid_minutes = price_counts > 0
    avg_prices = np.divide(price_sums, price_counts, out=np.zeros_like(price_sums), where=valid_minutes)
    if 'has_volume' in locals() and has_volume:
        vol_valid = volume_counts > 0
        avg_volumes = np.divide(volume_sums, volume_counts, out=np.zeros_like(volume_sums), where=vol_valid)
    
    # Conversion finale vers HH:MM (seulement pour l'affichage)
    result_data = []
    for minute_idx in np.where(valid_minutes)[0]:
        hour = minute_idx // 60
        minute = minute_idx % 60
        hhmm = f"{hour:02d}:{minute:02d}"
        row = {
            "hhmm": hhmm,
            "avg_price": float(avg_prices[minute_idx]),
            "count": int(price_counts[minute_idx])
        }
        if 'has_volume' in locals() and has_volume and volume_counts[minute_idx] > 0:
            row["avg_volume"] = float(avg_volumes[minute_idx])
        result_data.append(row)
    
    df_result = pd.DataFrame(result_data)
    
    metadata = {
        "symbol": symbol,
        "asset_class": asset_class,
        "price_col": price_col,
        "timezone": local_tz,
        "session_cfg": session_cfg,
        "total_observations": int(total_rows),
        "unique_minutes": int(valid_minutes.sum()),
        "has_volume": bool('has_volume' in locals() and has_volume)
    }
    
    return df_result, metadata


def plot_average_day(df: pd.DataFrame, metadata: Dict, title: Optional[str] = None) -> go.Figure:
    """Trace le graphique de l'average day avec Plotly, avec ticks horaires visibles
    et délimitation ETH/RTH (et breaks) quand la session le permet."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Aucune donnée", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Conversion HH:MM -> minute du jour pour un axe numérique lisible
    def _hhmm_to_min(hhmm: str) -> int:
        h, m = map(int, hhmm.split(":"))
        return h * 60 + m

    df = df.copy()
    df["minute_idx"] = df["hhmm"].apply(_hhmm_to_min)

    if title is None:
        symbol = metadata.get("symbol", "Unknown")
        price_col = metadata.get("price_col", "price")
        tz = metadata.get("timezone", "UTC")
        title = f"Average Day - {symbol} ({price_col}) - TZ: {tz}"
    
    fig = go.Figure()
    
    # Ligne principale (hover inclut l'heure et le nombre d'observations)
    fig.add_trace(go.Scatter(
        x=df["minute_idx"],
        y=df["avg_price"],
        mode='lines+markers',
        name=f'Avg {metadata.get("price_col", "price")}',
        line=dict(color='blue', width=2),
        marker=dict(size=4),
        text=df["hhmm"],
        customdata=df["count"],
        hovertemplate="Heure: %{text}<br>Avg " + metadata.get("price_col", "price") + ": %{y:.3f}<br>Observations: %{customdata}<extra></extra>"
    ))
    
    # Volume moyen si présent (utilise l'axe Y2 à droite)
    if "avg_volume" in df.columns:
        fig.add_trace(go.Bar(
            x=df["minute_idx"],
            y=df["avg_volume"],
            name='Volume moyen',
            yaxis='y2',
            opacity=0.25,
            marker_color='orange',
            customdata=df["hhmm"],
            hovertemplate="Heure: %{customdata}<br>Volume moyen: %{y}<extra></extra>"
        ))
    
    # Layout
    # Axe X: ticks horaires bien visibles (toutes les heures)
    hourly_ticks = [h * 60 for h in range(0, 24)]
    hourly_labels = [f"{h:02d}:00" for h in range(0, 24)]

    fig.update_layout(
        title=title,
        xaxis_title="Heure (TZ locale)",
        yaxis_title=f"Prix moyen ({metadata.get('price_col', 'price')})",
        yaxis2=dict(
            title="Volume moyen",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        hovermode='x unified',
        height=600,
        showlegend=True
    )
    fig.update_xaxes(
        tickmode='array',
        tickvals=hourly_ticks,
        ticktext=hourly_labels,
        range=[0, 1439],
        showgrid=True,
        tickangle=0
    )

    # Délimitation ETH/RTH/breaks si disponible
    sess = None
    session_cfg = metadata.get("session_cfg")
    if session_cfg and "session" in session_cfg:
        sess = session_cfg["session"]

    def _time_to_min(val: Optional[str]) -> Optional[int]:
        try:
            return _hhmm_to_min(val) if val else None
        except Exception:
            return None

    if sess:
        open_min = _time_to_min(sess.get("open"))
        close_min = _time_to_min(sess.get("close"))
        rth_open_min = _time_to_min(sess.get("rth_open"))
        rth_close_min = _time_to_min(sess.get("rth_close"))
        breaks = sess.get("breaks", []) or []

        # Helper pour ajouter un rectangle vertical [x0,x1)
        def add_vrect(x0: int, x1: int, color: str, label: Optional[str] = None, opacity: float = 0.08, line_color: str = "#bbbbbb"):
            if x0 is None or x1 is None:
                return
            # Normaliser bornes
            x0n = max(0, min(1439, x0))
            x1n = max(0, min(1439, x1))
            if x1n <= x0n:
                return
            fig.add_vrect(
                x0=x0n,
                x1=x1n,
                fillcolor=color,
                opacity=opacity,
                line_width=1,
                line_color=line_color,
                layer='below'
            )
            if label:
                fig.add_annotation(
                    x=(x0n + x1n) / 2,
                    y=1.02,
                    xref='x', yref='paper',
                    text=label,
                    showarrow=False,
                    font=dict(size=10, color='#444')
                )

        # RTH
        if rth_open_min is not None and rth_close_min is not None:
            # RTH en jaune (plus distinct du bleu ETH)
            add_vrect(rth_open_min, rth_close_min, '#FFD54F', 'RTH', opacity=0.06)

        # ETH: zones hors RTH à l'intérieur de la session complète
        if open_min is not None and close_min is not None and rth_open_min is not None and rth_close_min is not None:
            if open_min <= close_min:
                # Session diurne simple
                add_vrect(open_min, rth_open_min, '#1f77b4', 'ETH', opacity=0.06)
                add_vrect(rth_close_min, close_min, '#1f77b4', None, opacity=0.06)
            else:
                # Session overnight (ex: 19:00 -> 13:20)
                add_vrect(open_min, 1440, '#1f77b4', 'ETH', opacity=0.06)
                add_vrect(0, rth_open_min, '#1f77b4', None, opacity=0.06)
                add_vrect(rth_close_min, close_min, '#1f77b4', None, opacity=0.06)

        # Breaks
        for br in breaks:
            b0 = _time_to_min(br.get('start'))
            b1 = _time_to_min(br.get('end'))
            if b0 is not None and b1 is not None:
                # Break en blanc, semi-transparent, avec fin contour gris
                add_vrect(b0, b1, '#ffffff', 'Break', opacity=0.25, line_color='#bbbbbb')
    
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
