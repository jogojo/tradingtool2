"""
Configuration de l'application Trading Tool Pro
"""
import os
from pathlib import Path

# Chemins de base
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"

# Créer les dossiers s'ils n'existent pas
DATA_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

# Configuration des données
DATA_CONFIG = {
    "storage_format": "parquet",
    "partition_by": "year",
    "compression": "snappy",
    "cache_size_gb": 10
}

# Configuration First Rate Data
FIRST_RATE_CONFIG = {
    "api_endpoint": "https://api.firstratedata.com",
    "api_key": os.getenv("FIRST_RATE_API_KEY", ""),
    "default_frequency": "1min",
    "max_retries": 3
}

# Configuration des marchés
MARKETS_CONFIG = {
    "us_stocks": {
        "name": "US Stocks",
        "timezone": "America/New_York",
        "rth_open": "09:30",
        "rth_close": "16:00",
        "eth_open": "04:00",
        "eth_close": "20:00",
        "start_date": "2000-01-01"
    },
    "us_futures": {
        "name": "US Futures",
        "timezone": "America/New_York",
        "rth_open": "09:30",
        "rth_close": "16:00",
        "eth_open": "00:00",
        "eth_close": "23:59",
        "start_date": "2000-01-01"
    },
    "european_stocks": {
        "name": "European Stocks",
        "timezone": "Europe/Paris",
        "rth_open": "09:00",
        "rth_close": "17:30",
        "eth_open": "08:00",
        "eth_close": "18:00",
        "start_date": "2000-01-01"
    },
    "forex": {
        "name": "Forex",
        "timezone": "UTC",
        "rth_open": "00:00",
        "rth_close": "23:59",
        "eth_open": "00:00",
        "eth_close": "23:59",
        "start_date": "2000-01-01"
    },
    "crypto": {
        "name": "Crypto",
        "timezone": "UTC",
        "rth_open": "00:00",
        "rth_close": "23:59",
        "eth_open": "00:00",
        "eth_close": "23:59",
        "start_date": "2010-01-01"
    }
}

# Configuration IBKR
IBKR_CONFIG = {
    "host": "localhost",
    "port": 7497,
    "client_id": 1,
    "auto_connect": False
}

# Configuration de l'interface
UI_CONFIG = {
    "theme": "light",
    "page_title": "Trading Tool Pro",
    "page_icon": "📈",
    "layout": "wide"
}

# Configuration des analyses
ANALYSIS_CONFIG = {
    "periods": ["20y", "10y", "5y", "1y", "6m", "3m"],
    "frequencies": ["1min", "5min", "15min", "1h", "daily"],
    "auto_calculate": True
}
