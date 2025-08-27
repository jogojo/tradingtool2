from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
import sys

import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import logging

# Utiliser zoneinfo si Python 3.9+, sinon pytz
if sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo
    def get_timezone(tz_name: str):
        return ZoneInfo(tz_name)
    UTC = ZoneInfo("UTC")
else:
    import pytz
    def get_timezone(tz_name: str):
        return pytz.timezone(tz_name)
    UTC = pytz.UTC

from src.calendar.session_loader import TradingSessionTemplates
from src.silver.gap_fill import GapFiller

logger = logging.getLogger(__name__)

def _bronze_path(base_dir: Path, asset_type: str, symbol: str) -> Path:
    return base_dir / "bronze" / f"asset_class={asset_type}" / f"symbol={symbol}"


def _silver_base_path(base_dir: Path, asset_type: str, symbol: str) -> Path:
    return base_dir / "silver" / f"asset_class={asset_type}" / f"symbol={symbol}"


def _daily_path(base_dir: Path, asset_type: str, symbol: str) -> Path:
    return base_dir / "daily" / f"asset_class={asset_type}" / f"symbol={symbol}"


def run_gap_fill_for_symbol(
    base_data_dir: str,
    asset_type: str,
    symbol: str,
    session_template: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """
    Remplit les gaps minute pour un symbole sur une plage de dates en s'appuyant sur la base daily.
    - start_date/end_date: 'YYYY-MM-DD' en timezone du template (bornes inclusives). Si None: plage couverte par la daily.
    Écrit dans data/silver/asset_class=.../symbol=.../year=YYYY/data.parquet (UN fichier par année)
    """
    base = Path(base_data_dir)

    # Lire la daily pour connaître les jours à traiter
    daily_dir = _daily_path(base, asset_type, symbol)
    if not daily_dir.exists():
        raise FileNotFoundError(f"Daily introuvable pour {symbol} ({asset_type}) : {daily_dir}")
    daily_ds = ds.dataset(daily_dir, format="parquet")
    daily_tbl = daily_ds.to_table()
    daily_df = daily_tbl.to_pandas()
    if daily_df.empty:
        return {"processed_days": 0, "written_files": 0}

    # Déterminer la plage
    if start_date:
        daily_df = daily_df[daily_df["date"] >= pd.to_datetime(start_date).date()]
    if end_date:
        daily_df = daily_df[daily_df["date"] <= pd.to_datetime(end_date).date()]
    trading_days = sorted(daily_df["date"].unique())

    # Préparer session/gap filler
    sessions = TradingSessionTemplates()
    gap = GapFiller(sessions)
    tpl = sessions.get(session_template)
    tz = get_timezone(tpl["timezone"])  # local tz

    # Dataset minute source (bronze)
    bronze_dir = _bronze_path(base, asset_type, symbol)
    if not bronze_dir.exists():
        raise FileNotFoundError(f"Bronze introuvable pour {symbol} ({asset_type}) : {bronze_dir}")
    minute_ds = ds.dataset(bronze_dir, format="parquet")

    # Déterminer le premier jour disponible en Bronze (UTC → local date)
    # Objectif: ne pas tenter de remplir des jours Daily plus anciens que les données intraday disponibles.
    year_dirs = []
    try:
        year_dirs = sorted(
            [p for p in Path(bronze_dir).iterdir() if p.is_dir() and p.name.startswith("year=")],
            key=lambda p: int(p.name.split("=")[1])
        )
    except Exception:
        year_dirs = []

    first_bronze_ts_utc = None
    for ydir in year_dirs:
        try:
            ds_year = ds.dataset(ydir, format="parquet")
            tbl_year = ds_year.to_table(columns=["timestamp"])  # ne charger que la colonne timestamp
            if tbl_year.num_rows == 0:
                continue
            # Utiliser pandas pour conserver le tz
            ts_min = tbl_year.to_pandas()["timestamp"].min()
            if pd.notna(ts_min):
                ts_min = pd.to_datetime(ts_min)
                # Normaliser en tz-aware UTC
                if not pd.api.types.is_datetime64tz_dtype(pd.Series([ts_min])):
                    ts_min = pd.to_datetime(ts_min).tz_localize(UTC)
                else:
                    ts_min = pd.to_datetime(ts_min).tz_convert(UTC)
                first_bronze_ts_utc = ts_min
                break
        except Exception:
            continue

    if first_bronze_ts_utc is None:
        # Aucune donnée intraday disponible en bronze → rien à remplir
        return {"processed_days": 0, "written_files": 0}

    # Convertir en date locale (timezone du template)
    first_local_date = first_bronze_ts_utc.astimezone(tz).date()

    # Filtrer les trading_days pour commencer au premier jour où on a des données intraday
    trading_days = [d for d in trading_days if d >= first_local_date]
    if not trading_days:
        return {"processed_days": 0, "written_files": 0}

    # Accumulateur temporaire
    by_year: Dict[str, List[pd.DataFrame]] = {}

    for d in trading_days:
        if sys.version_info >= (3, 9):
            local_day = datetime.combine(d, datetime.min.time()).replace(tzinfo=tz)
        else:
            local_day = tz.localize(datetime.combine(d, datetime.min.time()))
        
        # Construire fenêtre UTC large (jour précédent 18:00 -> jour+1 18:00) pour capturer sessions chevauchantes
        start_local = local_day - timedelta(hours=12)
        end_local = local_day + timedelta(hours=36)
        start_utc = start_local.astimezone(UTC)
        end_utc = end_local.astimezone(UTC)

        # CORRECTION: Filtre Arrow avec type timestamp explicite
        ts_col = ds.field("timestamp")
        try:
            # Cast vers timestamp UTC pour éviter les surprises de type
            start_scalar = pa.scalar(start_utc, type=pa.timestamp('us', tz='UTC'))
            end_scalar = pa.scalar(end_utc, type=pa.timestamp('us', tz='UTC'))
            tbl = minute_ds.to_table(filter=(ts_col >= start_scalar) & (ts_col <= end_scalar))
        except Exception:
            tbl = minute_ds.to_table()
        df_min = tbl.to_pandas()
        if df_min.empty:
            continue

        # Vérification stricte: ne pas forcer; journaliser et lever une erreur explicite
        if pd.api.types.is_datetime64_any_dtype(df_min["timestamp"]) and not pd.api.types.is_datetime64tz_dtype(df_min["timestamp"]):
            sample = df_min["timestamp"].head(5).tolist()
            logger.error(
                "Bronze tz-naive détecté pour %s/%s le jour %s. Fenêtre UTC: %s → %s. Exemples: %s",
                asset_type,
                symbol,
                d,
                start_utc,
                end_utc,
                sample,
            )
            raise ValueError(
                "Minute data tz-naive détectée dans Bronze (voir logs). Corrige l'ingestion à la source."
            )

        filled = gap.fill_minutes(df_min, local_day, session_template)
        # Ajouter le contexte d'identité
        filled["asset_class"] = asset_type
        filled["symbol"] = symbol

        # CORRECTION: Ne pas utiliser year du premier timestamp uniquement - groupby par timestamp
        if not filled.empty:
            by_year.setdefault("temp", []).append(filled)

    # CORRECTION: Concaténer d'abord, puis groupby year(timestamp) pour écriture
    if not by_year.get("temp"):
        return {"processed_days": len(trading_days), "written_files": 0}
    
    all_data = pd.concat(by_year["temp"], ignore_index=True)
    
    # CORRECTION: Cast proper des types timestamp avant écriture Parquet (avec logs détaillés)
    try:
        all_data["timestamp"] = pd.to_datetime(all_data["timestamp"]).dt.tz_convert(UTC)
    except Exception as e:
        logger.error("Echec tz_convert sur 'timestamp'. dtype=%s, head=%s, erreur=%s",
                     all_data["timestamp"].dtype,
                     all_data["timestamp"].head(5).tolist(),
                     e)
        raise
    if "filled_from_ts" in all_data.columns:
        try:
            all_data["filled_from_ts"] = pd.to_datetime(all_data["filled_from_ts"]).dt.tz_convert(UTC)
        except Exception as e:
            logger.error("Echec tz_convert sur 'filled_from_ts'. dtype=%s, head=%s, erreur=%s",
                         all_data["filled_from_ts"].dtype,
                         all_data["filled_from_ts"].head(5).tolist(),
                         e)
            raise
    
    # Groupby year pour écriture par année
    all_data["year"] = all_data["timestamp"].dt.year
    years_groups = all_data.groupby("year")
    
    # Écriture: UN fichier par année
    base_out = _silver_base_path(base, asset_type, symbol)
    base_out.mkdir(parents=True, exist_ok=True)

    written_years = 0
    for year, df_y in years_groups:
        # Supprimer la colonne temporaire year
        df_y = df_y.drop("year", axis=1)
        
        # CORRECTION: Force les types avant conversion Arrow
        df_y["timestamp"] = df_y["timestamp"].astype("datetime64[ns, UTC]")
        if "filled_from_ts" in df_y.columns:
            df_y["filled_from_ts"] = df_y["filled_from_ts"].astype("datetime64[ns, UTC]")
        
        # Nettoyage du répertoire year
        year_dir = base_out / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        for f in year_dir.glob("*.parquet"):
            try:
                f.unlink()
            except Exception:
                pass
        # Écrire un seul parquet
        pq.write_table(pa.Table.from_pandas(df_y), year_dir / "bars_1min.parquet", compression="zstd")
        written_years += 1

    return {"processed_days": len(trading_days), "written_files": written_years}
