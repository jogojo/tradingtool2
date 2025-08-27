from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Optional, List
import sys

import pandas as pd

# Utiliser zoneinfo si Python 3.9+, sinon pytz
if sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo
    def get_timezone(tz_name: str):
        return ZoneInfo(tz_name)
else:
    import pytz
    def get_timezone(tz_name: str):
        return pytz.timezone(tz_name)

from src.calendar.session_loader import TradingSessionTemplates


class GapFiller:
    def __init__(self, session_templates: Optional[TradingSessionTemplates] = None) -> None:
        self.sessions = session_templates or TradingSessionTemplates()

    def generate_session_minutes(self, trading_date: datetime, session_cfg: dict) -> pd.DatetimeIndex:
        tz = get_timezone(session_cfg["timezone"])  # local TZ
        sess = session_cfg["session"]
        open_t = time.fromisoformat(sess["open"])  # e.g., 09:30
        close_t = time.fromisoformat(sess["close"])  # e.g., 16:00
        breaks = sess.get("breaks", [])

        # Gestion correcte des sessions overnight:
        # - Si open > close (ex: 18:00 -> 17:00), l'ouverture est la VEILLE et la clôture reste au jour J
        # - Sinon, open/close sont le même jour J
        if sys.version_info >= (3, 9):
            if open_t > close_t:
                local_open = datetime.combine(trading_date.date() - timedelta(days=1), open_t).replace(tzinfo=tz)
                local_close = datetime.combine(trading_date.date(), close_t).replace(tzinfo=tz)
            else:
                local_open = datetime.combine(trading_date.date(), open_t).replace(tzinfo=tz)
                local_close = datetime.combine(trading_date.date(), close_t).replace(tzinfo=tz)
        else:
            if open_t > close_t:
                local_open = tz.localize(datetime.combine(trading_date.date() - timedelta(days=1), open_t))
                local_close = tz.localize(datetime.combine(trading_date.date(), close_t))
            else:
                local_open = tz.localize(datetime.combine(trading_date.date(), open_t))
                local_close = tz.localize(datetime.combine(trading_date.date(), close_t))

        # CORRECTION: inclusive="left" pour [open, close) - fin exclue
        full_range = pd.date_range(local_open, local_close, freq="1min", inclusive="left")
        
        # Exclure les breaks [start, end) - fin de break exclue
        mask = pd.Series(True, index=full_range)
        for br in breaks:
            if sys.version_info >= (3, 9):
                br_start = datetime.combine(trading_date.date(), time.fromisoformat(br["start"])).replace(tzinfo=tz)
                br_end = datetime.combine(trading_date.date(), time.fromisoformat(br["end"])).replace(tzinfo=tz)
            else:
                br_start = tz.localize(datetime.combine(trading_date.date(), time.fromisoformat(br["start"])))
                br_end = tz.localize(datetime.combine(trading_date.date(), time.fromisoformat(br["end"])))
            
            if br_end <= br_start:
                br_end += timedelta(days=1)
            # CORRECTION: break [start, end) - fin exclue
            mask[(full_range >= br_start) & (full_range < br_end)] = False
        return full_range[mask]

    def fill_minutes(self, df_minute: pd.DataFrame, trading_date: datetime, session_name: str) -> pd.DataFrame:
        """
        df_minute: colonnes attendues au minimum [timestamp(UTC tz-aware), open, high, low, close, volume]
        trading_date: date de trading locale (dans le fuseau du template)
        session_name: nom du template (doit exister dans config)
        Retourne un DataFrame avec les minutes de la session; lignes manquantes remplies avec O=H=L=C=last_close, volume=0, filled_from_ts=ts d'origine.
        """
        tpl = self.sessions.get(session_name)
        tz = get_timezone(tpl["timezone"])  # local tz

        # Générer la grille de minutes en local puis convertir vers UTC
        local_minutes = self.generate_session_minutes(trading_date, tpl)
        # Logs de diagnostic des bornes de session
        import logging
        logging.getLogger(__name__).info(
            "Session locale %s: start=%s end=%s (size=%d)",
            session_name,
            local_minutes[0] if len(local_minutes) else None,
            local_minutes[-1] if len(local_minutes) else None,
            len(local_minutes),
        )
        if sys.version_info >= (3, 9):
            utc_minutes = local_minutes.tz_convert(ZoneInfo("UTC"))
        else:
            utc_minutes = local_minutes.tz_convert(pytz.UTC)
        logging.getLogger(__name__).info(
            "Session UTC %s: start=%s end=%s (size=%d)",
            session_name,
            utc_minutes[0] if len(utc_minutes) else None,
            utc_minutes[-1] if len(utc_minutes) else None,
            len(utc_minutes),
        )

        # Préparer les données d'origine indexées par minute UTC
        df = df_minute.copy()
        if not pd.api.types.is_datetime64tz_dtype(df["timestamp"]):
            # Log de diagnostic avant de lever l'erreur
            import logging
            logging.getLogger(__name__).error(
                "df_minute.timestamp tz-naive détecté dans fill_minutes. dtype=%s, head=%s",
                df["timestamp"].dtype,
                df["timestamp"].head(5).tolist(),
            )
            raise ValueError("df_minute.timestamp doit être timezone-aware (UTC)")
        df = df.sort_values("timestamp").set_index("timestamp")

        # Reindex sur la grille de minutes de session
        out = df.reindex(utc_minutes)

        # Marquer les lignes manquantes
        out["filled"] = out["close"].isna()
        # Sauvegarder la provenance (forcer dtype tz-aware UTC)
        try:
            out["filled_from_ts"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
        except Exception:
            # fallback pour versions pandas anciennes
            out["filled_from_ts"] = pd.NaT

        # CORRECTION: Pour les minutes synthétiques, O=H=L=C=last_close
        # D'abord propager seulement le close
        out["close"] = out["close"].ffill()
        
        # Pour les lignes remplies, mettre O=H=L=C (pas de propagation des high/low)
        last_close = out["close"]
        out.loc[out["filled"], "open"] = last_close.loc[out["filled"]]
        out.loc[out["filled"], "high"] = last_close.loc[out["filled"]]
        out.loc[out["filled"], "low"] = last_close.loc[out["filled"]]
        
        # Pour les lignes réelles, propager normalement O/H/L si manquantes
        out[["open", "high", "low"]] = out[["open", "high", "low"]].ffill()
        
        # Volume 0 pour lignes remplies
        out.loc[out["filled"], "volume"] = 0
        out["volume"] = out["volume"].fillna(0)  # Au cas où volume réel manquant
        
        # filled_from_ts = dernière timestamp non nulle précédente
        prev_real_ts = out.index.to_series().where(~out["filled"]).ffill()
        # IMPORTANT: ne pas utiliser .values (qui droppe le tz). Assigner la série directement.
        try:
            out.loc[out["filled"], "filled_from_ts"] = prev_real_ts.loc[out["filled"]]
        except Exception:
            # Alignement explicite si nécessaire
            to_assign = prev_real_ts.loc[out["filled"]]
            to_assign.index = out.index[out["filled"]]
            out.loc[out["filled"], "filled_from_ts"] = to_assign

        out = out.reset_index().rename(columns={"index": "timestamp"})
        return out
