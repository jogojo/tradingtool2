import argparse
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

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


def _get_period_bounds(period_type: str, current_date: datetime = None) -> Tuple[datetime, datetime]:
    """Calcule les bornes de période selon le type demandé."""
    if current_date is None:
        current_date = datetime.now()
    
    if period_type == "toute_periode":
        # Sera géré par compute_avgday_all_period (sans filtre temporel)
        return None, None
    elif period_type == "dernier_mois":
        end_date = current_date
        start_date = end_date - timedelta(days=30)
    elif period_type == "6_derniers_mois":
        end_date = current_date
        start_date = end_date - timedelta(days=180)
    elif period_type == "3_dernieres_annees":
        end_date = current_date
        start_date = end_date - timedelta(days=3*365)
    else:
        # Format "4ans_YYYY" pour les périodes de 4 ans historiques
        if period_type.startswith("4ans_"):
            year_end = int(period_type.split("_")[1])
            end_date = datetime(year_end, 12, 31)
            start_date = datetime(year_end - 3, 1, 1)  # 4 ans: -3 années
        else:
            raise ValueError(f"Type de période non supporté: {period_type}")
    
    return start_date, end_date


def _get_available_periods(base_dir: Path, asset_class: str, symbol: str) -> List[str]:
    """Détermine les périodes disponibles selon les données présentes."""
    base_sym = base_dir / "silver" / f"asset_class={asset_class}" / f"symbol={symbol}"
    if not base_sym.exists():
        return []
    
    # Trouver l'étendue des données
    dset = ds.dataset(base_sym, format="parquet")
    try:
        scanner = dset.scanner(columns=["timestamp"])
        ts_min = None
        ts_max = None
        
        for batch_group in scanner.scan_batches():
            try:
                batch = batch_group.to_record_batch()
            except Exception:
                batch = getattr(batch_group, "record_batch", batch_group)
            
            if batch.num_rows == 0:
                continue
                
            ts_col = batch.column(0)
            try:
                cur_min = pc.min(ts_col).as_py()
                cur_max = pc.max(ts_col).as_py()
                if cur_min: ts_min = cur_min if ts_min is None or cur_min < ts_min else ts_min
                if cur_max: ts_max = cur_max if ts_max is None or cur_max > ts_max else ts_max
            except Exception:
                pass
        
        if ts_min is None or ts_max is None:
            return []
            
        # Convertir en datetime
        if hasattr(ts_min, 'to_pydatetime'):
            ts_min = ts_min.to_pydatetime()
        if hasattr(ts_max, 'to_pydatetime'):
            ts_max = ts_max.to_pydatetime()
            
        current_date = datetime.now()
        periods = ["toute_periode", "dernier_mois", "6_derniers_mois", "3_dernieres_annees"]
        
        # Ajouter les périodes de 4 ans historiques
        current_year = current_date.year
        four_years_ago = current_year - 4
        
        # Remonter par tranches de 4 ans jusqu'au début des données
        start_year = ts_min.year
        for end_year in range(four_years_ago, start_year - 1, -4):
            if end_year >= start_year + 3:  # Au moins 4 ans de données
                periods.append(f"4ans_{end_year}")
        
        return periods
        
    except Exception:
        return ["toute_periode", "dernier_mois", "6_derniers_mois", "3_dernieres_annees"]


def compute_avgday_all_period(base_dir: Path, asset_class: str, symbol: str, price_col: str = "close") -> Tuple[pd.DataFrame, Dict]:
    """
    Calcule l'average day sur TOUTE la période disponible dans la base.
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
    ts_min_global = None
    ts_max_global = None
    
    # Stream optimisé par batches SANS filtre temporel (toute la période)
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
            vol_idx = batch.schema.get_field_index("volume")
            if vol_idx != -1:
                vol_col_idx = batch.column(vol_idx)
        
        try:
            # Trouver min/max global pour metadata
            try:
                cur_min = pc.min(ts_col).as_py()
                cur_max = pc.max(ts_col).as_py()
                if cur_min: ts_min_global = cur_min if ts_min_global is None or cur_min < ts_min_global else ts_min_global
                if cur_max: ts_max_global = cur_max if ts_max_global is None or cur_max > ts_max_global else ts_max_global
            except Exception:
                pass
            
            # Conversion timezone UNE SEULE FOIS par batch
            if local_tz == "UTC":
                ts_local = ts_col
            else:
                ts_local = pc.assume_timezone(ts_col, timezone="UTC")
                ts_local = pc.cast(ts_local, pa.timestamp("ns", tz=local_tz))
            
            # Extraction heure/minute en entiers
            hour = pc.hour(ts_local)
            minute = pc.minute(ts_local)
            
            # CLEF: minute_du_jour = hour * 60 + minute (0-1439)
            minute_of_day = pc.add(pc.multiply(hour, 60), minute)
            
            # Filtre session si défini
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
                
                # Accumulation vectorisée
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
            # Fallback pandas si Arrow échoue
            df_batch = batch.to_pandas()
            df_batch = df_batch.dropna()
            if df_batch.empty:
                continue
                
            df_batch["timestamp"] = pd.to_datetime(df_batch["timestamp"], utc=True)
            
            # Min/max global
            try:
                batch_min = df_batch["timestamp"].min()
                batch_max = df_batch["timestamp"].max()
                if ts_min_global is None or batch_min < ts_min_global:
                    ts_min_global = batch_min
                if ts_max_global is None or batch_max > ts_max_global:
                    ts_max_global = batch_max
            except Exception:
                pass
            
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
                prices_pd = df_batch[price_col].iloc[mask.values]
                vols_pd = df_batch["volume"].iloc[mask.values] if has_volume and "volume" in df_batch.columns else None
            else:
                prices_pd = df_batch[price_col]
                vols_pd = df_batch["volume"] if has_volume and "volume" in df_batch.columns else None
            
            # Vectorisation numpy
            minute_idx = minute_of_day.values.astype(np.int32)
            prices_vals = prices_pd.values
            
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
    if has_volume:
        vol_valid = volume_counts > 0
        avg_volumes = np.divide(volume_sums, volume_counts, out=np.zeros_like(volume_sums), where=vol_valid)
    
    # Conversion finale vers HH:MM
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
        if has_volume and volume_counts[minute_idx] > 0:
            row["avg_volume"] = float(avg_volumes[minute_idx])
        result_data.append(row)
    
    df_result = pd.DataFrame(result_data)
    
    # Metadata avec période complète
    start_date = ts_min_global.isoformat() if ts_min_global else "Inconnue"
    end_date = ts_max_global.isoformat() if ts_max_global else "Inconnue"
    
    metadata = {
        "symbol": symbol,
        "asset_class": asset_class,
        "price_col": price_col,
        "timezone": local_tz,
        "session_cfg": session_cfg,
        "period_type": "toute_periode",
        "start_date": start_date,
        "end_date": end_date,
        "total_observations": int(total_rows),
        "unique_minutes": int(valid_minutes.sum()),
        "has_volume": bool(has_volume)
    }
    
    return df_result, metadata


def compute_avgday_by_period(base_dir: Path, asset_class: str, symbol: str, period_type: str, price_col: str = "close") -> Tuple[pd.DataFrame, Dict]:
    """
    Calcule l'average day pour une période spécifique en utilisant la méthodologie ultra-rapide.
    """
    # Si c'est "toute_periode", utiliser la fonction spécialisée
    if period_type == "toute_periode":
        return compute_avgday_all_period(base_dir, asset_class, symbol, price_col)
    
    import numpy as np
    
    base_sym = base_dir / "silver" / f"asset_class={asset_class}" / f"symbol={symbol}"
    if not base_sym.exists():
        return pd.DataFrame(columns=["hhmm", "avg_price", "count"]), {"error": "Symbole non trouvé"}

    # Résolution timezone + session (sans fallback; retourne erreur explicite)
    try:
        local_tz, session_cfg = _resolve_session_timezone(symbol)
    except Exception as e:
        return pd.DataFrame(columns=["hhmm", "avg_price", "count"]), {"error": str(e)}

    # Calcul des bornes de période
    try:
        start_date, end_date = _get_period_bounds(period_type)
        if start_date is None or end_date is None:
            return pd.DataFrame(columns=["hhmm", "avg_price", "count"]), {"error": "Période non définie"}
        start_ts = pd.Timestamp(start_date, tz='UTC')
        end_ts = pd.Timestamp(end_date, tz='UTC')
    except Exception as e:
        return pd.DataFrame(columns=["hhmm", "avg_price", "count"]), {"error": f"Erreur période: {str(e)}"}
    
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
    
    # Stream optimisé par batches avec filtre temporel
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
            vol_idx = batch.schema.get_field_index("volume")
            if vol_idx != -1:
                vol_col_idx = batch.column(vol_idx)
        
        try:
            # FILTRE TEMPOREL: ne garder que la période demandée
            ts_filter = pc.and_(
                pc.greater_equal(ts_col, start_ts.value),
                pc.less_equal(ts_col, end_ts.value)
            )
            
            # Appliquer le filtre temporel
            if pc.any(ts_filter).as_py():
                ts_col = pc.filter(ts_col, ts_filter)
                price_col_idx = pc.filter(price_col_idx, ts_filter)
                if vol_col_idx is not None:
                    vol_col_idx = pc.filter(vol_col_idx, ts_filter)
            else:
                continue  # Aucune donnée dans cette période pour ce batch
            
            if ts_col.length == 0:
                continue
            
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
            # Fallback pandas si Arrow échoue
            df_batch = batch.to_pandas()
            df_batch = df_batch.dropna()
            if df_batch.empty:
                continue
                
            df_batch["timestamp"] = pd.to_datetime(df_batch["timestamp"], utc=True)
            
            # Filtre temporel pandas
            mask_period = (df_batch["timestamp"] >= start_ts) & (df_batch["timestamp"] <= end_ts)
            df_batch = df_batch[mask_period]
            if df_batch.empty:
                continue
            
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
                prices_pd = df_batch[price_col].iloc[mask.values]
                vols_pd = df_batch["volume"].iloc[mask.values] if has_volume and "volume" in df_batch.columns else None
            else:
                prices_pd = df_batch[price_col]
                vols_pd = df_batch["volume"] if has_volume and "volume" in df_batch.columns else None
            
            # Vectorisation numpy
            minute_idx = minute_of_day.values.astype(np.int32)
            prices_vals = prices_pd.values
            
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
    if has_volume:
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
        if has_volume and volume_counts[minute_idx] > 0:
            row["avg_volume"] = float(avg_volumes[minute_idx])
        result_data.append(row)
    
    df_result = pd.DataFrame(result_data)
    
    print(f"DEBUG {period_type} FINAL: {len(df_result)} minutes, {total_rows} obs totales, {valid_minutes.sum()} minutes uniques")
    
    metadata = {
        "symbol": symbol,
        "asset_class": asset_class,
        "price_col": price_col,
        "timezone": local_tz,
        "session_cfg": session_cfg,
        "period_type": period_type,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "total_observations": int(total_rows),
        "unique_minutes": int(valid_minutes.sum()),
        "has_volume": bool(has_volume)
    }
    
    return df_result, metadata


def compute_avgday_by_weekday(base_dir: Path, asset_class: str, symbol: str, price_col: str = "close") -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """
    Calcule l'average day par jour de la semaine pour les 3 dernières années.
    Retourne un dictionnaire {jour: DataFrame}.
    """
    import numpy as np
    
    base_sym = base_dir / "silver" / f"asset_class={asset_class}" / f"symbol={symbol}"
    if not base_sym.exists():
        return {}, {"error": "Symbole non trouvé"}

    # Résolution timezone + session
    try:
        local_tz, session_cfg = _resolve_session_timezone(symbol)
    except Exception as e:
        return {}, {"error": str(e)}

    # Période: 3 dernières années
    current_date = datetime.now()
    start_date = current_date - timedelta(days=3*365)
    end_date = current_date
    start_ts = pd.Timestamp(start_date, tz='UTC')
    end_ts = pd.Timestamp(end_date, tz='UTC')
    
    # Dataset ultra-minimal
    dset = ds.dataset(base_sym, format="parquet")
    required_cols = ["timestamp", price_col]
    if "volume" in dset.schema.names:
        required_cols.append("volume")
    cols = [c for c in required_cols if c in dset.schema.names]
    if price_col not in cols:
        return {}, {"error": f"Colonne {price_col} manquante"}

    scanner = dset.scanner(columns=cols)
    
    # Session info
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

    # Accumulateurs par jour de la semaine (0=Monday, 6=Sunday)
    weekdays = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    weekday_data = {}
    for wd in weekdays:
        weekday_data[wd] = {
            "price_sums": np.zeros(1440, dtype=np.float64),
            "price_counts": np.zeros(1440, dtype=np.int64),
            "volume_sums": np.zeros(1440, dtype=np.float64) if "volume" in cols else None,
            "volume_counts": np.zeros(1440, dtype=np.int64) if "volume" in cols else None,
            "total_rows": 0
        }
    
    has_volume = "volume" in cols
    
    # Stream par batches avec analyse par jour de semaine
    for batch_group in scanner.scan_batches():
        try:
            batch = batch_group.to_record_batch()
        except Exception:
            batch = getattr(batch_group, "record_batch", batch_group)
        
        if batch.num_rows == 0:
            continue
            
        ts_col = batch.column(0)
        price_col_idx = batch.column(1)
        vol_col_idx = None
        if has_volume:
            vol_idx = batch.schema.get_field_index("volume")
            if vol_idx != -1:
                vol_col_idx = batch.column(vol_idx)
        
        try:
            # Filtre temporel (3 dernières années)
            ts_filter = pc.and_(
                pc.greater_equal(ts_col, start_ts.value),
                pc.less_equal(ts_col, end_ts.value)
            )
            
            if not pc.any(ts_filter).as_py():
                continue
                
            ts_col = pc.filter(ts_col, ts_filter)
            price_col_idx = pc.filter(price_col_idx, ts_filter)
            if vol_col_idx is not None:
                vol_col_idx = pc.filter(vol_col_idx, ts_filter)
            
            if ts_col.length == 0:
                continue
            
            # Conversion timezone
            if local_tz == "UTC":
                ts_local = ts_col
            else:
                ts_local = pc.assume_timezone(ts_col, timezone="UTC")
                ts_local = pc.cast(ts_local, pa.timestamp("ns", tz=local_tz))
            
            # Extraction jour de semaine + heure/minute
            weekday = pc.day_of_week(ts_local)  # 0=Sunday, 1=Monday, ..., 6=Saturday
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
                
                weekday = pc.filter(weekday, session_mask)
                minute_of_day = pc.filter(minute_of_day, session_mask)
                price_col_idx = pc.filter(price_col_idx, session_mask)
                if vol_col_idx is not None:
                    vol_col_idx = pc.filter(vol_col_idx, session_mask)
            
            # Conversion numpy
            weekday_np = weekday.to_numpy()
            minute_idx = minute_of_day.to_numpy()
            prices = price_col_idx.to_numpy()
            
            # Ajustement: Arrow donne 0=Sunday, on veut 0=Monday
            weekday_np = (weekday_np + 6) % 7  # Maintenant 0=Monday, 6=Sunday
            
            # Filtres
            valid_mask = ~(np.isnan(prices) | np.isnan(minute_idx.astype(float)))
            weekday_np = weekday_np[valid_mask].astype(np.int32)
            minute_idx = minute_idx[valid_mask].astype(np.int32)
            prices = prices[valid_mask]
            
            valid_minutes = (minute_idx >= 0) & (minute_idx < 1440) & (weekday_np >= 0) & (weekday_np < 7)
            weekday_np = weekday_np[valid_minutes]
            minute_idx = minute_idx[valid_minutes]
            prices = prices[valid_minutes]
            
            # Traitement du volume
            vols = None
            if vol_col_idx is not None:
                vols = vol_col_idx.to_numpy()[valid_mask][valid_minutes]
            
            # Accumuler par jour de semaine
            for wd_idx in range(7):
                wd_mask = weekday_np == wd_idx
                if not wd_mask.any():
                    continue
                    
                wd_minutes = minute_idx[wd_mask]
                wd_prices = prices[wd_mask]
                
                if len(wd_minutes) > 0:
                    wd_name = weekdays[wd_idx]
                    batch_sums = np.bincount(wd_minutes, weights=wd_prices, minlength=1440)
                    batch_counts = np.bincount(wd_minutes, minlength=1440)
                    
                    weekday_data[wd_name]["price_sums"] += batch_sums
                    weekday_data[wd_name]["price_counts"] += batch_counts
                    weekday_data[wd_name]["total_rows"] += len(wd_minutes)
                    
                    # Volume
                    if vols is not None and weekday_data[wd_name]["volume_sums"] is not None:
                        wd_vols = vols[wd_mask]
                        vol_valid = ~np.isnan(wd_vols)
                        if vol_valid.any():
                            v_sums = np.bincount(wd_minutes[vol_valid], weights=wd_vols[vol_valid], minlength=1440)
                            v_counts = np.bincount(wd_minutes[vol_valid], minlength=1440)
                            weekday_data[wd_name]["volume_sums"] += v_sums
                            weekday_data[wd_name]["volume_counts"] += v_counts
            
        except Exception:
            # Fallback pandas
            df_batch = batch.to_pandas()
            df_batch = df_batch.dropna()
            if df_batch.empty:
                continue
                
            df_batch["timestamp"] = pd.to_datetime(df_batch["timestamp"], utc=True)
            
            # Filtre temporel
            mask_period = (df_batch["timestamp"] >= start_ts) & (df_batch["timestamp"] <= end_ts)
            df_batch = df_batch[mask_period]
            if df_batch.empty:
                continue
            
            # Conversion timezone
            if local_tz == "UTC":
                ts_local = df_batch["timestamp"]
            else:
                try:
                    ts_local = df_batch["timestamp"].dt.tz_convert(local_tz)
                except Exception:
                    ts_local = df_batch["timestamp"]
            
            # Jour de semaine et minute du jour
            weekday_pd = ts_local.dt.dayofweek  # 0=Monday
            minute_of_day = ts_local.dt.hour * 60 + ts_local.dt.minute
            
            # Filtre session
            if session_start_min is not None and session_end_min is not None:
                if session_start_min <= session_end_min:
                    mask = (minute_of_day >= session_start_min) & (minute_of_day < session_end_min)
                else:
                    mask = (minute_of_day >= session_start_min) | (minute_of_day < session_end_min)
                
                weekday_pd = weekday_pd[mask]
                minute_of_day = minute_of_day[mask]
                prices_pd = df_batch[price_col].iloc[mask.values]
                vols_pd = df_batch["volume"].iloc[mask.values] if has_volume and "volume" in df_batch.columns else None
            else:
                prices_pd = df_batch[price_col]
                vols_pd = df_batch["volume"] if has_volume and "volume" in df_batch.columns else None
            
            # Traitement par jour de semaine
            for wd_idx in range(7):
                wd_mask = weekday_pd == wd_idx
                if not wd_mask.any():
                    continue
                    
                wd_minutes = minute_of_day[wd_mask].values.astype(np.int32)
                wd_prices = prices_pd[wd_mask].values
                
                valid_mask = ~(np.isnan(wd_prices) | (wd_minutes < 0) | (wd_minutes >= 1440))
                wd_minutes = wd_minutes[valid_mask]
                wd_prices = wd_prices[valid_mask]
                
                if len(wd_minutes) > 0:
                    wd_name = weekdays[wd_idx]
                    batch_sums = np.bincount(wd_minutes, weights=wd_prices, minlength=1440)
                    batch_counts = np.bincount(wd_minutes, minlength=1440)
                    
                    weekday_data[wd_name]["price_sums"] += batch_sums
                    weekday_data[wd_name]["price_counts"] += batch_counts
                    weekday_data[wd_name]["total_rows"] += len(wd_minutes)
                    
                    # Volume
                    if vols_pd is not None and weekday_data[wd_name]["volume_sums"] is not None:
                        wd_vols = vols_pd[wd_mask].values[valid_mask]
                        vol_valid = ~np.isnan(wd_vols)
                        if vol_valid.any():
                            v_sums = np.bincount(wd_minutes[vol_valid], weights=wd_vols[vol_valid], minlength=1440)
                            v_counts = np.bincount(wd_minutes[vol_valid], minlength=1440)
                            weekday_data[wd_name]["volume_sums"] += v_sums
                            weekday_data[wd_name]["volume_counts"] += v_counts
    
    # Conversion finale en DataFrames
    result_dataframes = {}
    total_observations = 0
    
    for wd_name, data in weekday_data.items():
        valid_minutes = data["price_counts"] > 0
        if not valid_minutes.any():
            result_dataframes[wd_name] = pd.DataFrame(columns=["hhmm", "avg_price", "count"])
            continue
            
        avg_prices = np.divide(data["price_sums"], data["price_counts"], 
                              out=np.zeros_like(data["price_sums"]), where=valid_minutes)
        
        result_data = []
        for minute_idx in np.where(valid_minutes)[0]:
            hour = minute_idx // 60
            minute = minute_idx % 60
            hhmm = f"{hour:02d}:{minute:02d}"
            row = {
                "hhmm": hhmm,
                "avg_price": float(avg_prices[minute_idx]),
                "count": int(data["price_counts"][minute_idx])
            }
            
            if has_volume and data["volume_sums"] is not None and data["volume_counts"][minute_idx] > 0:
                avg_vol = data["volume_sums"][minute_idx] / data["volume_counts"][minute_idx]
                row["avg_volume"] = float(avg_vol)
                
            result_data.append(row)
        
        result_dataframes[wd_name] = pd.DataFrame(result_data)
        total_observations += data["total_rows"]
    
    metadata = {
        "symbol": symbol,
        "asset_class": asset_class,
        "price_col": price_col,
        "timezone": local_tz,
        "session_cfg": session_cfg,
        "period_type": "3_dernieres_annees_par_weekday",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "total_observations": int(total_observations),
        "has_volume": bool(has_volume)
    }
    
    return result_dataframes, metadata


def plot_avgday_periods(periods_data: Dict[str, pd.DataFrame], metadata: Dict, title: Optional[str] = None) -> go.Figure:
    """Trace les average days par périodes avec zones RTH/ETH/Breaks et volume."""
    if not periods_data:
        fig = go.Figure()
        fig.add_annotation(text="Aucune donnée", x=0.5, y=0.5, showarrow=False)
        return fig
    
    def _hhmm_to_min(hhmm: str) -> int:
        h, m = map(int, hhmm.split(":"))
        return h * 60 + m

    if title is None:
        symbol = metadata.get("symbol", "Unknown")
        price_col = metadata.get("price_col", "price")
        tz = metadata.get("timezone", "UTC")
        title = f"Average Day par Périodes - {symbol} ({price_col}) - TZ: {tz}"
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22']
    has_volume = False
    
    # Lignes principales (prix)
    for i, (period, df) in enumerate(periods_data.items()):
        if df.empty:
            continue
            
        df = df.copy()
        df["minute_idx"] = df["hhmm"].apply(_hhmm_to_min)
        # CRUCIAL: Trier par minute_idx pour avoir des lignes continues
        df = df.sort_values("minute_idx")
        
        # Vérifier volume
        if "avg_volume" in df.columns and not has_volume:
            has_volume = True
        
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=df["minute_idx"],
            y=df["avg_price"],
            mode='lines',
            name=period,
            line=dict(color=color, width=2),
            text=df["hhmm"],
            customdata=df["count"],
            hovertemplate=f"<b>{period}</b><br>Heure: %{{text}}<br>Prix moyen: %{{y:.3f}}<br>Observations: %{{customdata}}<extra></extra>"
        ))
    
    # Volume moyen si présent (axe Y3 à droite)
    if has_volume:
        for i, (period, df) in enumerate(periods_data.items()):
            if df.empty or "avg_volume" not in df.columns:
                continue
                
            df = df.copy()
            df["minute_idx"] = df["hhmm"].apply(_hhmm_to_min)
            df = df.sort_values("minute_idx")
            
            color = colors[i % len(colors)]
            fig.add_trace(go.Bar(
                x=df["minute_idx"],
                y=df["avg_volume"],
                name=f'Volume {period}',
                yaxis='y2',
                opacity=0.25,
                marker_color=color,
                customdata=df["hhmm"],
                hovertemplate=f"<b>{period}</b><br>Heure: %{{customdata}}<br>Volume moyen: %{{y}}<extra></extra>"
            ))
    
    # Configuration des axes
    hourly_ticks = [h * 60 for h in range(0, 24)]
    hourly_labels = [f"{h:02d}:00" for h in range(0, 24)]
    
    layout_config = {
        "title": title,
        "xaxis_title": "Heure (TZ locale)",
        "yaxis_title": f"Prix moyen ({metadata.get('price_col', 'price')})",
        "hovermode": 'x unified',
        "height": 600,
        "showlegend": True
    }
    
    if has_volume:
        layout_config["yaxis2"] = dict(
            title="Volume moyen",
            overlaying='y',
            side='right',
            showgrid=False
        )
    
    fig.update_layout(**layout_config)
    fig.update_xaxes(
        tickmode='array',
        tickvals=hourly_ticks,
        ticktext=hourly_labels,
        range=[0, 1439],
        showgrid=True,
        tickangle=0
    )

    # Délimitation RTH/ETH/Breaks (comme dans average_day.py)
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

        # RTH en jaune
        if rth_open_min is not None and rth_close_min is not None:
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

        # Breaks en blanc
        for br in breaks:
            b0 = _time_to_min(br.get('start'))
            b1 = _time_to_min(br.get('end'))
            if b0 is not None and b1 is not None:
                add_vrect(b0, b1, '#ffffff', 'Break', opacity=0.25, line_color='#bbbbbb')
    
    return fig


def plot_avgday_weekdays(weekdays_data: Dict[str, pd.DataFrame], metadata: Dict, title: Optional[str] = None) -> go.Figure:
    """Trace les average days par jour de semaine avec zones RTH/ETH/Breaks et volume."""
    if not weekdays_data:
        fig = go.Figure()
        fig.add_annotation(text="Aucune donnée", x=0.5, y=0.5, showarrow=False)
        return fig
    
    def _hhmm_to_min(hhmm: str) -> int:
        h, m = map(int, hhmm.split(":"))
        return h * 60 + m

    if title is None:
        symbol = metadata.get("symbol", "Unknown")
        price_col = metadata.get("price_col", "price")
        tz = metadata.get("timezone", "UTC")
        title = f"Average Day par Jour de Semaine - {symbol} ({price_col}) - TZ: {tz}"
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    weekday_order = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    has_volume = False
    
    # Lignes principales (prix)
    for i, weekday in enumerate(weekday_order):
        if weekday not in weekdays_data or weekdays_data[weekday].empty:
            continue
            
        df = weekdays_data[weekday].copy()
        df["minute_idx"] = df["hhmm"].apply(_hhmm_to_min)
        # CRUCIAL: Trier par minute_idx pour avoir des lignes continues
        df = df.sort_values("minute_idx")
        
        # Vérifier volume
        if "avg_volume" in df.columns and not has_volume:
            has_volume = True
        
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=df["minute_idx"],
            y=df["avg_price"],
            mode='lines',
            name=weekday,
            line=dict(color=color, width=2),
            text=df["hhmm"],
            customdata=df["count"],
            hovertemplate=f"<b>{weekday}</b><br>Heure: %{{text}}<br>Prix moyen: %{{y:.3f}}<br>Observations: %{{customdata}}<extra></extra>"
        ))
    
    # Volume moyen si présent (axe Y2 à droite)
    if has_volume:
        for i, weekday in enumerate(weekday_order):
            if weekday not in weekdays_data or weekdays_data[weekday].empty:
                continue
            
            df = weekdays_data[weekday].copy()
            if "avg_volume" not in df.columns:
                continue
                
            df["minute_idx"] = df["hhmm"].apply(_hhmm_to_min)
            df = df.sort_values("minute_idx")
            
            color = colors[i % len(colors)]
            fig.add_trace(go.Bar(
                x=df["minute_idx"],
                y=df["avg_volume"],
                name=f'Volume {weekday}',
                yaxis='y2',
                opacity=0.25,
                marker_color=color,
                customdata=df["hhmm"],
                hovertemplate=f"<b>{weekday}</b><br>Heure: %{{customdata}}<br>Volume moyen: %{{y}}<extra></extra>"
            ))
    
    # Configuration des axes
    hourly_ticks = [h * 60 for h in range(0, 24)]
    hourly_labels = [f"{h:02d}:00" for h in range(0, 24)]
    
    layout_config = {
        "title": title,
        "xaxis_title": "Heure (TZ locale)",
        "yaxis_title": f"Prix moyen ({metadata.get('price_col', 'price')})",
        "hovermode": 'x unified',
        "height": 600,
        "showlegend": True
    }
    
    if has_volume:
        layout_config["yaxis2"] = dict(
            title="Volume moyen",
            overlaying='y',
            side='right',
            showgrid=False
        )
    
    fig.update_layout(**layout_config)
    fig.update_xaxes(
        tickmode='array',
        tickvals=hourly_ticks,
        ticktext=hourly_labels,
        range=[0, 1439],
        showgrid=True,
        tickangle=0
    )

    # Délimitation RTH/ETH/Breaks (comme dans average_day.py)
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

        # RTH en jaune
        if rth_open_min is not None and rth_close_min is not None:
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

        # Breaks en blanc
        for br in breaks:
            b0 = _time_to_min(br.get('start'))
            b1 = _time_to_min(br.get('end'))
            if b0 is not None and b1 is not None:
                add_vrect(b0, b1, '#ffffff', 'Break', opacity=0.25, line_color='#bbbbbb')
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Calcul Average Day par périodes et jours de semaine")
    parser.add_argument("--base", type=str, default="./data")
    parser.add_argument("--asset", type=str, required=True)
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--price", type=str, default="close", help="Colonne de prix")
    parser.add_argument("--period", type=str, help="Période spécifique (dernier_mois, 6_derniers_mois, etc.)")
    parser.add_argument("--weekdays", action="store_true", help="Analyser par jour de semaine")
    parser.add_argument("--plot", action="store_true", help="Afficher le graphique")
    args = parser.parse_args()

    base_path = Path(args.base)
    
    if args.weekdays:
        weekdays_data, metadata = compute_avgday_by_weekday(base_path, args.asset, args.symbol, args.price)
        print("== Metadata ==")
        for k, v in metadata.items():
            if k != "session_cfg":
                print(f"{k}: {v}")
        
        for weekday, df in weekdays_data.items():
            print(f"\n== {weekday} ==")
            print(df.to_string(index=False))
        
        if args.plot:
            fig = plot_avgday_weekdays(weekdays_data, metadata)
            fig.show()
    
    elif args.period:
        df, metadata = compute_avgday_by_period(base_path, args.asset, args.symbol, args.period, args.price)
        print("== Metadata ==")
        for k, v in metadata.items():
            if k != "session_cfg":
                print(f"{k}: {v}")
        
        print(f"\n== {args.period} ==")
        print(df.to_string(index=False))
    
    else:
        # Analyser toutes les périodes disponibles
        available_periods = _get_available_periods(base_path, args.asset, args.symbol)
        periods_data = {}
        
        for period in available_periods:
            df, metadata = compute_avgday_by_period(base_path, args.asset, args.symbol, period, args.price)
            if not df.empty:
                periods_data[period] = df
        
        print("== Périodes disponibles ==")
        for period in periods_data.keys():
            print(f"- {period}")
        
        if args.plot and periods_data:
            fig = plot_avgday_periods(periods_data, metadata)
            fig.show()


if __name__ == "__main__":
    main()
