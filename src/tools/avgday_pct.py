import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import plotly.graph_objects as go

from src.calendar.session_loader import TradingSessionTemplates, SymbolSessionRegistry


def _resolve_session_timezone(symbol: str) -> Tuple[str, Dict]:
    templates = TradingSessionTemplates()
    reg = SymbolSessionRegistry()
    tpl_name = reg.get(symbol)
    if not tpl_name:
        raise ValueError(f"Aucune règle de session trouvée pour '{symbol}'. Mappez le symbole dans la page Calendriers.")
    if tpl_name not in templates.templates:
        raise ValueError(f"Template de session '{tpl_name}' introuvable pour '{symbol}'. Vérifiez config/trading_sessions.json.")
    return templates.templates[tpl_name].get("timezone", "UTC"), templates.templates[tpl_name]


def _scan_prices_fast(base_dir: Path, asset_class: str, symbol: str, price_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    ULTRA-RAPIDE: Scan 100% vectorisé qui retourne numpy arrays directement.
    Returns: (minute_idx[N], prices[N], date_ids[N], metadata)
    """
    base_sym = base_dir / "silver" / f"asset_class={asset_class}" / f"symbol={symbol}"
    if not base_sym.exists():
        return np.array([]), np.array([]), np.array([]), {"error": "Symbole non trouvé"}

    local_tz, session_cfg = _resolve_session_timezone(symbol)

    dset = ds.dataset(base_sym, format="parquet")
    required_cols = ["timestamp", price_col]
    cols = [c for c in required_cols if c in dset.schema.names]
    if price_col not in cols:
        return np.array([]), np.array([]), np.array([]), {"error": f"Colonne {price_col} manquante"}

    scanner = dset.scanner(columns=cols)

    # Session bounds en minutes
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

    # Accumulateurs numpy (pas de DataFrames intermédiaires)
    all_minute_idx = []
    all_prices = []
    all_date_ids = []
    
    for batch_group in scanner.scan_batches():
        try:
            batch = batch_group.to_record_batch()
        except Exception:
            batch = getattr(batch_group, "record_batch", batch_group)
        if batch.num_rows == 0:
            continue

        ts_col = batch.column(0)
        price_col_arrow = batch.column(1)

        try:
            # CORRECTION 1: Conversion timezone CORRECTE
            if local_tz == "UTC":
                ts_local = ts_col
            else:
                # Étape 1: assume UTC (ré-étiquetage)
                ts_utc = pc.assume_timezone(ts_col, timezone="UTC")
                # Étape 2: conversion vers timezone locale
                ts_local = pc.cast(ts_utc, pa.timestamp("ns", tz=local_tz))

            # Extraction heure/minute AVANT filtrage session
            hour = pc.hour(ts_local)
            minute = pc.minute(ts_local)
            minute_of_day = pc.add(pc.multiply(hour, 60), minute)
            
            # AMÉLIORATION 4: Filtre session sur Arrow AVANT conversion numpy
            if session_start_min is not None and session_end_min is not None:
                if session_start_min <= session_end_min:
                    session_mask = pc.and_(
                        pc.greater_equal(minute_of_day, session_start_min),
                        pc.less(minute_of_day, session_end_min)
                    )
                else:
                    session_mask = pc.or_(
                        pc.greater_equal(minute_of_day, session_start_min),
                        pc.less(minute_of_day, session_end_min)
                    )
                
                # Appliquer filtre session
                ts_local = pc.filter(ts_local, session_mask)
                price_col_arrow = pc.filter(price_col_arrow, session_mask)
                minute_of_day = pc.filter(minute_of_day, session_mask)
            
            if ts_local.length == 0:
                continue

            # AMÉLIORATION 5: Date ID entier (epoch days) au lieu d'objets date
            # Extraire date directement depuis timestamp local
            date_arrow = pc.date(ts_local)  # Date Arrow
            # Convertir en "jours depuis epoch" (int32)
            epoch_date = pa.scalar(pa.date32().encode(datetime(1970, 1, 1).date()))
            date_ids = pc.days_between(epoch_date, date_arrow)
            
            # AMÉLIORATION 2: Conversion numpy DIRECTE (pas pandas intermédiaire)
            minute_idx = minute_of_day.to_numpy().astype(np.int32)
            prices = price_col_arrow.to_numpy().astype(np.float64)
            date_ids_np = date_ids.to_numpy().astype(np.int32)
            
            # Filtre NaN + minutes valides
            valid_mask = ~np.isnan(prices) & (minute_idx >= 0) & (minute_idx < 1440)
            minute_idx = minute_idx[valid_mask]
            prices = prices[valid_mask]
            date_ids_np = date_ids_np[valid_mask]
            
            # AMÉLIORATION 3: Accumulation arrays numpy (pas DataFrames)
            if len(minute_idx) > 0:
                all_minute_idx.append(minute_idx)
                all_prices.append(prices)
                all_date_ids.append(date_ids_np)
                
        except Exception:
            # Fallback ultra-minimal si Arrow échoue
            df_batch = batch.to_pandas()
            if df_batch.empty:
                continue
                
            # Conversion timezone correcte
            df_batch["timestamp"] = pd.to_datetime(df_batch["timestamp"], utc=True)
            if local_tz != "UTC":
                try:
                    df_batch["timestamp"] = df_batch["timestamp"].dt.tz_convert(local_tz)
                except Exception:
                    pass
            
            # Date ID + minute
            epoch = pd.Timestamp('1970-01-01').tz_localize(df_batch["timestamp"].dt.tz)
            date_ids_pd = (df_batch["timestamp"].dt.normalize() - epoch).dt.days.astype(np.int32)
            minute_idx_pd = (df_batch["timestamp"].dt.hour * 60 + df_batch["timestamp"].dt.minute).astype(np.int32)
            prices_pd = df_batch[price_col].astype(np.float64)
            
            # Filtre session
            if session_start_min is not None and session_end_min is not None:
                if session_start_min <= session_end_min:
                    mask = (minute_idx_pd >= session_start_min) & (minute_idx_pd < session_end_min)
                else:
                    mask = (minute_idx_pd >= session_start_min) | (minute_idx_pd < session_end_min)
                minute_idx_pd = minute_idx_pd[mask]
                prices_pd = prices_pd[mask]
                date_ids_pd = date_ids_pd[mask]
            
            # Filtre valides
            valid_mask = ~np.isnan(prices_pd) & (minute_idx_pd >= 0) & (minute_idx_pd < 1440)
            minute_idx_pd = minute_idx_pd[valid_mask]
            prices_pd = prices_pd[valid_mask]
            date_ids_pd = date_ids_pd[valid_mask]
            
            if len(minute_idx_pd) > 0:
                all_minute_idx.append(minute_idx_pd.values)
                all_prices.append(prices_pd.values)
                all_date_ids.append(date_ids_pd.values)

    # AMÉLIORATION 3: Concatenation numpy finale (une seule fois)
    if not all_minute_idx:
        return np.array([]), np.array([]), np.array([]), {"error": "Aucune donnée"}

    minute_idx_final = np.concatenate(all_minute_idx)
    prices_final = np.concatenate(all_prices)
    date_ids_final = np.concatenate(all_date_ids)

    metadata = {
        "timezone": local_tz,
        "session_cfg": session_cfg,
        "price_col": price_col,
    }
    return minute_idx_final, prices_final, date_ids_final, metadata


def compute_average_day_pct_period(base_dir: Path, asset_class: str, symbol: str, start_date: datetime, end_date: datetime, price_col: str = "close") -> Tuple[pd.DataFrame, Dict]:
    """
    ULTRA-RAPIDE: Calcul % pour une période spécifique [start_date, end_date].
    """
    minute_idx, prices, date_ids, meta = _scan_prices_fast_period(base_dir, asset_class, symbol, price_col, start_date, end_date)
    if "error" in meta:
        return pd.DataFrame(columns=["hhmm", "avg_pct", "count"]), meta

    if len(prices) == 0:
        return pd.DataFrame(columns=["hhmm", "avg_pct", "count"]), {"error": "Aucune donnée pour cette période"}

    # Tri et calcul % identique
    sort_idx = np.lexsort((minute_idx, date_ids))
    minute_idx = minute_idx[sort_idx]
    prices = prices[sort_idx]
    date_ids = date_ids[sort_idx]

    unique_date_ids, session_starts = np.unique(date_ids, return_index=True)
    session_counts = np.diff(np.append(session_starts, len(prices)))
    
    # CORRECTION: Prix d'ouverture = prix à la PREMIÈRE MINUTE DE SESSION
    session_open_minute = None
    if meta.get("session_cfg") and "session" in meta["session_cfg"]:
        try:
            from datetime import time
            sess = meta["session_cfg"]["session"]
            start_time = time.fromisoformat(sess["open"])
            session_open_minute = start_time.hour * 60 + start_time.minute
        except Exception:
            pass
    
    # Pour chaque session, trouver le prix à la minute d'ouverture
    open_prices = np.zeros(len(unique_date_ids))
    
    for i, session_start_idx in enumerate(session_starts):
        session_end_idx = session_start_idx + session_counts[i]
        session_minutes = minute_idx[session_start_idx:session_end_idx]
        session_prices = prices[session_start_idx:session_end_idx]
        
        if session_open_minute is not None:
            # Chercher la minute d'ouverture exacte
            open_mask = session_minutes == session_open_minute
            if np.any(open_mask):
                open_prices[i] = session_prices[open_mask][0]
            else:
                # Fallback: minute la plus proche de l'ouverture
                closest_idx = np.argmin(np.abs(session_minutes - session_open_minute))
                open_prices[i] = session_prices[closest_idx]
        else:
            # Fallback: premier prix chronologique
            open_prices[i] = session_prices[0]
    
    open_broadcast = np.repeat(open_prices, session_counts)

    valid_mask = (open_broadcast != 0) & ~np.isnan(prices)
    pct_changes = np.zeros_like(prices)
    pct_changes[valid_mask] = ((prices[valid_mask] / open_broadcast[valid_mask]) - 1.0) * 100.0

    pct_sums = np.bincount(minute_idx, weights=pct_changes, minlength=1440)
    counts = np.bincount(minute_idx, minlength=1440)
    avg_pct = np.divide(pct_sums, counts, out=np.zeros_like(pct_sums), where=counts > 0)

    result = []
    for minute in range(1440):
        if counts[minute] > 0:
            hh = minute // 60
            mm = minute % 60
            result.append({
                "hhmm": f"{hh:02d}:{mm:02d}",
                "avg_pct": float(avg_pct[minute]),
                "count": int(counts[minute])
            })

    df_result = pd.DataFrame(result)

    metadata = {
        "symbol": symbol,
        "asset_class": asset_class,
        "price_col": price_col,
        "timezone": meta.get("timezone"),
        "session_cfg": meta.get("session_cfg"),
        "total_observations": int(counts.sum()),
        "unique_minutes": int((counts > 0).sum()),
        "total_sessions": len(unique_date_ids),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat()
    }

    return df_result, metadata


def _scan_prices_fast_period(base_dir: Path, asset_class: str, symbol: str, price_col: str, start_date: datetime, end_date: datetime) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Scan avec filtre temporel [start_date, end_date].
    """
    base_sym = base_dir / "silver" / f"asset_class={asset_class}" / f"symbol={symbol}"
    if not base_sym.exists():
        return np.array([]), np.array([]), np.array([]), {"error": "Symbole non trouvé"}

    local_tz, session_cfg = _resolve_session_timezone(symbol)

    dset = ds.dataset(base_sym, format="parquet")
    required_cols = ["timestamp", price_col]
    cols = [c for c in required_cols if c in dset.schema.names]
    if price_col not in cols:
        return np.array([]), np.array([]), np.array([]), {"error": f"Colonne {price_col} manquante"}

    scanner = dset.scanner(columns=cols)

    # Session bounds
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

    # Conversion dates en timestamps UTC pour filtre
    start_ts = pd.Timestamp(start_date, tz='UTC')
    end_ts = pd.Timestamp(end_date, tz='UTC')

    all_minute_idx = []
    all_prices = []
    all_date_ids = []
    
    for batch_group in scanner.scan_batches():
        try:
            batch = batch_group.to_record_batch()
        except Exception:
            batch = getattr(batch_group, "record_batch", batch_group)
        if batch.num_rows == 0:
            continue

        ts_col = batch.column(0)
        price_col_arrow = batch.column(1)

        try:
            # FILTRE TEMPOREL en premier
            ts_filter = pc.and_(
                pc.greater_equal(ts_col, start_ts.value),
                pc.less_equal(ts_col, end_ts.value)
            )
            
            if not pc.any(ts_filter).as_py():
                continue
                
            ts_col = pc.filter(ts_col, ts_filter)
            price_col_arrow = pc.filter(price_col_arrow, ts_filter)
            
            if ts_col.length == 0:
                continue

            # Conversion timezone correcte
            if local_tz == "UTC":
                ts_local = ts_col
            else:
                ts_utc = pc.assume_timezone(ts_col, timezone="UTC")
                ts_local = pc.cast(ts_utc, pa.timestamp("ns", tz=local_tz))

            hour = pc.hour(ts_local)
            minute = pc.minute(ts_local)
            minute_of_day = pc.add(pc.multiply(hour, 60), minute)
            
            # Filtre session
            if session_start_min is not None and session_end_min is not None:
                if session_start_min <= session_end_min:
                    session_mask = pc.and_(
                        pc.greater_equal(minute_of_day, session_start_min),
                        pc.less(minute_of_day, session_end_min)
                    )
                else:
                    session_mask = pc.or_(
                        pc.greater_equal(minute_of_day, session_start_min),
                        pc.less(minute_of_day, session_end_min)
                    )
                
                ts_local = pc.filter(ts_local, session_mask)
                price_col_arrow = pc.filter(price_col_arrow, session_mask)
                minute_of_day = pc.filter(minute_of_day, session_mask)
            
            if ts_local.length == 0:
                continue

            # Date ID entier
            date_arrow = pc.date(ts_local)
            epoch_date = pa.scalar(pa.date32().encode(datetime(1970, 1, 1).date()))
            date_ids = pc.days_between(epoch_date, date_arrow)
            
            minute_idx = minute_of_day.to_numpy().astype(np.int32)
            prices = price_col_arrow.to_numpy().astype(np.float64)
            date_ids_np = date_ids.to_numpy().astype(np.int32)
            
            valid_mask = ~np.isnan(prices) & (minute_idx >= 0) & (minute_idx < 1440)
            minute_idx = minute_idx[valid_mask]
            prices = prices[valid_mask]
            date_ids_np = date_ids_np[valid_mask]
            
            if len(minute_idx) > 0:
                all_minute_idx.append(minute_idx)
                all_prices.append(prices)
                all_date_ids.append(date_ids_np)
                
        except Exception:
            # Fallback pandas
            df_batch = batch.to_pandas()
            if df_batch.empty:
                continue
                
            df_batch["timestamp"] = pd.to_datetime(df_batch["timestamp"], utc=True)
            
            # Filtre temporel
            mask_period = (df_batch["timestamp"] >= start_ts) & (df_batch["timestamp"] <= end_ts)
            df_batch = df_batch[mask_period]
            if df_batch.empty:
                continue
            
            # Conversion timezone
            if local_tz != "UTC":
                try:
                    df_batch["timestamp"] = df_batch["timestamp"].dt.tz_convert(local_tz)
                except Exception:
                    pass
            
            epoch = pd.Timestamp('1970-01-01').tz_localize(df_batch["timestamp"].dt.tz)
            date_ids_pd = (df_batch["timestamp"].dt.normalize() - epoch).dt.days.astype(np.int32)
            minute_idx_pd = (df_batch["timestamp"].dt.hour * 60 + df_batch["timestamp"].dt.minute).astype(np.int32)
            prices_pd = df_batch[price_col].astype(np.float64)
            
            # Filtre session
            if session_start_min is not None and session_end_min is not None:
                if session_start_min <= session_end_min:
                    mask = (minute_idx_pd >= session_start_min) & (minute_idx_pd < session_end_min)
                else:
                    mask = (minute_idx_pd >= session_start_min) | (minute_idx_pd < session_end_min)
                minute_idx_pd = minute_idx_pd[mask]
                prices_pd = prices_pd[mask]
                date_ids_pd = date_ids_pd[mask]
            
            valid_mask = ~np.isnan(prices_pd) & (minute_idx_pd >= 0) & (minute_idx_pd < 1440)
            minute_idx_pd = minute_idx_pd[valid_mask]
            prices_pd = prices_pd[valid_mask]
            date_ids_pd = date_ids_pd[valid_mask]
            
            if len(minute_idx_pd) > 0:
                all_minute_idx.append(minute_idx_pd.values)
                all_prices.append(prices_pd.values)
                all_date_ids.append(date_ids_pd.values)

    if not all_minute_idx:
        return np.array([]), np.array([]), np.array([]), {"error": "Aucune donnée pour cette période"}

    minute_idx_final = np.concatenate(all_minute_idx)
    prices_final = np.concatenate(all_prices)
    date_ids_final = np.concatenate(all_date_ids)

    metadata = {
        "timezone": local_tz,
        "session_cfg": session_cfg,
        "price_col": price_col,
    }
    return minute_idx_final, prices_final, date_ids_final, metadata


def compute_average_day_pct(base_dir: Path, asset_class: str, symbol: str, price_col: str = "close") -> Tuple[pd.DataFrame, Dict]:
    """
    ULTRA-RAPIDE: Calcul 100% vectorisé en pourcentage vs ouverture session.
    """
    minute_idx, prices, date_ids, meta = _scan_prices_fast(base_dir, asset_class, symbol, price_col)
    if "error" in meta:
        return pd.DataFrame(columns=["hhmm", "avg_pct", "count"]), meta

    if len(prices) == 0:
        return pd.DataFrame(columns=["hhmm", "avg_pct", "count"]), {"error": "Aucune donnée"}

    # AMÉLIORATION 6: Tri par (date_id, minute_idx) ultra-rapide sur entiers
    sort_idx = np.lexsort((minute_idx, date_ids))
    minute_idx = minute_idx[sort_idx]
    prices = prices[sort_idx]
    date_ids = date_ids[sort_idx]

    # AMÉLIORATION 6: Sessions via date_ids entiers (pas objets date)
    unique_date_ids, session_starts = np.unique(date_ids, return_index=True)
    session_counts = np.diff(np.append(session_starts, len(prices)))
    
    # CORRECTION: Prix d'ouverture = prix à la PREMIÈRE MINUTE DE SESSION (pas première minute de journée)
    # Trouver la minute d'ouverture de session depuis config
    session_open_minute = None
    if meta.get("session_cfg") and "session" in meta["session_cfg"]:
        try:
            from datetime import time
            sess = meta["session_cfg"]["session"]
            start_time = time.fromisoformat(sess["open"])
            session_open_minute = start_time.hour * 60 + start_time.minute
        except Exception:
            pass
    
    # Pour chaque session, trouver le prix à la minute d'ouverture (ou le plus proche)
    open_prices = np.zeros(len(unique_date_ids))
    
    for i, session_start_idx in enumerate(session_starts):
        session_end_idx = session_start_idx + session_counts[i]
        session_minutes = minute_idx[session_start_idx:session_end_idx]
        session_prices = prices[session_start_idx:session_end_idx]
        
        if session_open_minute is not None:
            # Chercher la minute d'ouverture exacte dans cette session
            open_mask = session_minutes == session_open_minute
            if np.any(open_mask):
                # Prendre le premier prix à la minute d'ouverture
                open_prices[i] = session_prices[open_mask][0]
            else:
                # Fallback: minute la plus proche de l'ouverture
                closest_idx = np.argmin(np.abs(session_minutes - session_open_minute))
                open_prices[i] = session_prices[closest_idx]
        else:
            # Fallback: premier prix chronologique de la session
            open_prices[i] = session_prices[0]
    
    # Broadcast des prix d'ouverture sur toute la session
    open_broadcast = np.repeat(open_prices, session_counts)

    # Calcul % vectorisé
    valid_mask = (open_broadcast != 0) & ~np.isnan(prices)
    pct_changes = np.zeros_like(prices)
    pct_changes[valid_mask] = ((prices[valid_mask] / open_broadcast[valid_mask]) - 1.0) * 100.0

    # Agrégation par minute avec bincount (ultra-rapide)
    pct_sums = np.bincount(minute_idx, weights=pct_changes, minlength=1440)
    counts = np.bincount(minute_idx, minlength=1440)
    avg_pct = np.divide(pct_sums, counts, out=np.zeros_like(pct_sums), where=counts > 0)

    # AMÉLIORATION 2: Formatting HH:MM seulement à la fin
    result = []
    for minute in range(1440):
        if counts[minute] > 0:
            hh = minute // 60
            mm = minute % 60
            result.append({
                "hhmm": f"{hh:02d}:{mm:02d}",
                "avg_pct": float(avg_pct[minute]),
                "count": int(counts[minute])
            })

    df_result = pd.DataFrame(result)

    metadata = {
        "symbol": symbol,
        "asset_class": asset_class,
        "price_col": price_col,
        "timezone": meta.get("timezone"),
        "session_cfg": meta.get("session_cfg"),
        "total_observations": int(counts.sum()),
        "unique_minutes": int((counts > 0).sum()),
        "total_sessions": len(unique_date_ids)
    }

    return df_result, metadata


def plot_average_day_pct(df: pd.DataFrame, metadata: Dict, title: Optional[str] = None) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Aucune donnée", x=0.5, y=0.5, showarrow=False)
        return fig

    def _hhmm_to_min(hhmm: str) -> int:
        h, m = map(int, hhmm.split(":"))
        return h * 60 + m

    df = df.copy()
    df["minute_idx"] = df["hhmm"].apply(_hhmm_to_min)

    if title is None:
        tz = metadata.get("timezone", "UTC")
        title = f"Average Day % - TZ: {tz}"

    fig = go.Figure()

    # Courbe principale en %
    fig.add_trace(go.Scatter(
        x=df["minute_idx"],
        y=df["avg_pct"],
        mode='lines+markers',
        name='Avg %',
        line=dict(color='blue', width=2),
        marker=dict(size=4),
        text=df["hhmm"],
        customdata=df["count"],
        hovertemplate="Heure: %{text}<br>Performance: %{y:.3f}%<br>Observations: %{customdata}<extra></extra>"
    ))

    # Layout
    hourly_ticks = [h * 60 for h in range(0, 24)]
    hourly_labels = [f"{h:02d}:00" for h in range(0, 24)]

    fig.update_layout(
        title=title,
        xaxis_title="Heure (TZ locale)",
        yaxis_title="Performance moyenne (%)",
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

    # Zones RTH/ETH/Breaks
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

        def add_vrect(x0: int, x1: int, color: str, label: Optional[str] = None, opacity: float = 0.06, line_color: str = "#bbbbbb"):
            if x0 is None or x1 is None:
                return
            x0n = max(0, min(1439, x0))
            x1n = max(0, min(1439, x1))
            if x1n <= x0n:
                return
            fig.add_vrect(x0=x0n, x1=x1n, fillcolor=color, opacity=opacity, line_width=1, line_color=line_color, layer='below')

        if rth_open_min is not None and rth_close_min is not None:
            fig.add_vrect(x0=rth_open_min, x1=rth_close_min, fillcolor='#FFD54F', opacity=0.06, line_width=1, line_color='#bbbbbb')

        if open_min is not None and close_min is not None and rth_open_min is not None and rth_close_min is not None:
            if open_min <= close_min:
                add_vrect(open_min, rth_open_min, '#1f77b4', 'ETH')
                add_vrect(rth_close_min, close_min, '#1f77b4', None)
            else:
                add_vrect(open_min, 1440, '#1f77b4', 'ETH')
                add_vrect(0, rth_open_min, '#1f77b4', None)
                add_vrect(rth_close_min, close_min, '#1f77b4', None)

        for br in breaks:
            b0 = _time_to_min(br.get('start'))
            b1 = _time_to_min(br.get('end'))
            if b0 is not None and b1 is not None:
                fig.add_vrect(x0=b0, x1=b1, fillcolor='#ffffff', opacity=0.25, line_width=1, line_color='#bbbbbb')

    return fig


def main():
    parser = argparse.ArgumentParser(description="Average Day en % (ultra-rapide)")
    parser.add_argument("--base", type=str, default="./data")
    parser.add_argument("--asset", type=str, required=True)
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--price", type=str, default="close")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    df, metadata = compute_average_day_pct(Path(args.base), args.asset, args.symbol, args.price)
    print("== Metadata ==")
    for k, v in metadata.items():
        if k != "session_cfg":
            print(f"{k}: {v}")
    if args.plot and not df.empty:
        fig = plot_average_day_pct(df, metadata)
        fig.show()


if __name__ == "__main__":
    main()
