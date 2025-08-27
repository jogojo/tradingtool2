import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


def audit_parquet_file(file_path: Path) -> Dict[str, object]:
    """
    Audite un fichier Parquet individuel pour vérifier que la colonne 'timestamp'
    est bien de type Arrow timestamp avec timezone 'UTC'.

    Pour limiter l'I/O, on inspecte le schéma et on charge seulement le premier
    row group (colonne 'timestamp') si disponible.
    """
    result: Dict[str, object] = {
        "file": str(file_path),
        "has_timestamp": False,
        "arrow_tz": None,
        "arrow_ok": False,
        "row_groups": 0,
        "sample_checked": False,
        "issues": [],
    }

    try:
        pf = pq.ParquetFile(str(file_path))
        result["row_groups"] = pf.num_row_groups
        schema = pf.schema_arrow

        try:
            ts_field = schema.field("timestamp")
            result["has_timestamp"] = True
        except KeyError:
            result["issues"].append("missing_timestamp_column")
            return result

        ts_type = ts_field.type
        if not pa.types.is_timestamp(ts_type):
            result["issues"].append(f"timestamp_not_arrow_timestamp: {ts_type}")
            return result

        # Arrow timezone
        tz = ts_type.tz  # None ou string (e.g., 'UTC')
        result["arrow_tz"] = tz
        if tz != "UTC":
            result["issues"].append(f"timestamp_timezone_not_UTC: {tz}")
        else:
            result["arrow_ok"] = True

        # Lecture échantillon (premier row group seulement)
        if pf.num_row_groups > 0:
            try:
                col_tbl = pf.read_row_group(0, columns=["timestamp"])  # pyarrow.Table
                # Forcer conversion pandas pour détecter dtypes inattendus
                ts_series = col_tbl.to_pandas()["timestamp"]
                # pandas dtype tz-aware ?
                if not getattr(ts_series.dtype, "tz", None):
                    result["issues"].append("pandas_timestamp_tz_naive_in_sample")
                else:
                    if str(ts_series.dtype.tz) not in ("UTC", "tzutc()"):
                        result["issues"].append(f"pandas_timestamp_tz_not_UTC_in_sample: {ts_series.dtype.tz}")
                result["sample_checked"] = True
            except Exception as e:
                result["issues"].append(f"sample_read_error: {type(e).__name__}: {e}")
    except Exception as e:
        result["issues"].append(f"parquet_open_error: {type(e).__name__}: {e}")

    return result


def find_parquet_files(bronze_dir: Path) -> List[Path]:
    return list(bronze_dir.rglob("*.parquet"))


def audit_bronze(base_dir: Path) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """
    Audite l'ensemble de la base BRONZE et retourne:
      - la liste des résultats par fichier
      - un résumé des problèmes
    """
    bronze_dir = base_dir / "bronze"
    files = find_parquet_files(bronze_dir)
    results: List[Dict[str, object]] = []
    issues_summary: Dict[str, int] = {}

    for f in files:
        res = audit_parquet_file(f)
        results.append(res)
        for it in res["issues"]:
            issues_summary[it] = issues_summary.get(it, 0) + 1

    return results, issues_summary


def main():
    parser = argparse.ArgumentParser(description="Audit UTC awareness pour la base BRONZE")
    parser.add_argument("--base-dir", default="./data", help="Répertoire racine des données (par défaut ./data)")
    parser.add_argument("--verbose", action="store_true", help="Affiche les détails par fichier")
    args = parser.parse_args()

    base = Path(args.base_dir)
    bronze_dir = base / "bronze"
    if not bronze_dir.exists():
        print(f"❌ Répertoire BRONZE introuvable: {bronze_dir}")
        raise SystemExit(2)

    files = find_parquet_files(bronze_dir)
    if not files:
        print("⚠️  Aucun fichier Parquet trouvé dans BRONZE.")
        raise SystemExit(0)

    total = 0
    ok = 0
    issues_total = 0
    issues_summary: Dict[str, int] = {}

    for f in files:
        total += 1
        res = audit_parquet_file(f)

        if args.verbose:
            print(res)

        if res["arrow_ok"] and (not res["issues"]):
            ok += 1
        else:
            if not res["arrow_ok"]:
                issues_total += 1
            for it in res["issues"]:
                issues_summary[it] = issues_summary.get(it, 0) + 1

    print("\n=== Rapport d'audit BRONZE (UTC awareness) ===")
    print(f"Fichiers scannés : {total}")
    print(f"Fichiers OK      : {ok}")
    print(f"Fichiers à revoir: {issues_total}")
    if issues_summary:
        print("Détails des problèmes:")
        for k, v in sorted(issues_summary.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f" - {k}: {v}")

    # Code de sortie non-nul si problèmes détectés
    raise SystemExit(0 if issues_total == 0 else 1)


if __name__ == "__main__":
    main()


