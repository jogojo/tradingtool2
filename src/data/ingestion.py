"""
Module d'ingestion des données BRONZE
Traite les 6 types de sources avec formats bornés et partitionnement strict
"""
import os
import zipfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from pathlib import Path
from datetime import datetime
import pytz
from typing import Dict, List, Optional, Tuple
import logging
import re

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    """
    Classe principale pour l'ingestion BRONZE des données
    """
    
    # Pas de configuration de format - parsing direct selon le type d'asset
    
    # Colonnes standard pour tous les fichiers
    STANDARD_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Configuration des colonnes requises par asset_class
    COLUMN_REQUIREMENTS = {
        "stock": ["timestamp", "open", "high", "low", "close", "volume"],
        "etf": ["timestamp", "open", "high", "low", "close", "volume"],
        "future": ["timestamp", "open", "high", "low", "close", "volume"],
        "crypto": ["timestamp", "open", "high", "low", "close", "volume"],
        "forex": ["timestamp", "open", "high", "low", "close", "volume"],
        "index": ["timestamp", "open", "high", "low", "close", "volume"]
    }
    
    def __init__(self, base_data_dir: str = "./data"):
        self.base_data_dir = Path(base_data_dir)
        self.asset_types = ["crypto", "forex", "stock", "etf", "future", "index"]
        
        # Créer la structure BRONZE
        self.bronze_dir = self.base_data_dir / "bronze"
        for asset_type in self.asset_types:
            (self.bronze_dir / f"asset_class={asset_type}").mkdir(parents=True, exist_ok=True)
        
        # Timezone de référence (UTC pour le stockage)
        self.utc_tz = pytz.UTC
        self.eastern_tz = pytz.timezone('America/New_York')
        
        # Buffer pour regrouper les chunks avant écriture
        self.chunk_buffer = {}
        self.buffer_size_limit = 5000000  # 5M lignes par partition
    
    def process_file(self, file_path: str, asset_type: str) -> Dict[str, int]:
        """
        Traite automatiquement un fichier selon son type (ZIP, CSV, TXT)
        Détecte automatiquement le symbole depuis le nom du fichier
        """
        print(f"🚀 DÉBUT process_file - Fichier: {file_path}")
        print(f"🚀 DÉBUT process_file - Asset type: {asset_type}")
        
        logger.info(f"=== DÉBUT process_file ===")
        logger.info(f"Fichier: {file_path}")
        logger.info(f"Type d'asset: {asset_type}")
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"❌ Fichier non trouvé: {file_path}")
            logger.error(f"Fichier non trouvé: {file_path}")
            raise FileNotFoundError(f"Fichier non trouvé : {file_path}")
        
        if asset_type not in self.asset_types:
            print(f"❌ Type d'asset non valide: {asset_type}")
            logger.error(f"Type d'asset non valide: {asset_type}")
            raise ValueError(f"Type d'asset non valide : {asset_type}")
        
        # Détecter automatiquement le symbole
        symbol = self._detect_symbol_from_path(str(file_path), asset_type)
        print(f"✅ Symbole détecté: {symbol}")
        logger.info(f"Symbole détecté: {symbol}")
        
        # Détection automatique du type de fichier
        file_suffix = file_path.suffix.lower()
        print(f"📁 Extension du fichier: {file_suffix}")
        logger.info(f"Extension du fichier: {file_suffix}")
        
        if file_suffix == '.zip':
            print("📦 Traitement comme fichier ZIP")
            logger.info("Traitement comme fichier ZIP")
            return self.process_zip_file(str(file_path), asset_type, symbol)
        elif file_suffix in ['.csv', '.txt']:
            print("📄 Traitement comme fichier CSV/TXT")
            logger.info("Traitement comme fichier CSV/TXT")
            return self.process_csv_file(str(file_path), asset_type, symbol)
        else:
            print(f"❌ Type de fichier non supporté: {file_suffix}")
            logger.error(f"Type de fichier non supporté: {file_suffix}")
            raise ValueError(f"Type de fichier non supporté : {file_suffix}")
    
    def process_zip_file(self, zip_path: str, asset_type: str, symbol: str) -> Dict[str, int]:
        """
        Traite un fichier ZIP contenant des données
        """
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Fichier ZIP non trouvé : {zip_path}")
        
        results = {"fichiers_traités": 0, "lignes_traitées": 0, "erreurs": 0}
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                data_files = [f for f in zip_file.namelist() 
                            if f.endswith(('.csv', '.txt'))]
                
                for data_file in data_files:
                    try:
                        # Détecter le symbole pour chaque fichier dans le ZIP
                        file_symbol = self._detect_symbol_from_path(data_file, asset_type)
                        self._process_file_from_zip(zip_file, data_file, asset_type, file_symbol)
                        results["fichiers_traités"] += 1
                        
                    except Exception as e:
                        logger.error(f"Erreur lors du traitement de {data_file}: {e}")
                        results["erreurs"] += 1
                        
        except Exception as e:
            logger.error(f"Erreur lors de l'ouverture du ZIP {zip_path}: {e}")
            raise
        
        # Écrire les buffers restants
        self._flush_all_buffers()
        
        return results
    
    def _process_file_from_zip(self, zip_file: zipfile.ZipFile, file_name: str, asset_type: str, symbol: str):
        """
        Traite un fichier depuis un ZIP en streaming
        """
        with zip_file.open(file_name) as data_file:
            chunk_size = 500000  # 500K lignes par chunk
            
            # Choix des colonnes selon l'asset
            if asset_type == "forex":
                read_names = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
            else:
                read_names = self.STANDARD_COLUMNS
            
            for chunk_num, chunk_df in enumerate(pd.read_csv(
                data_file,
                chunksize=chunk_size,
                names=read_names,
                header=None,
                sep=','
            )):
                # Combiner date+time pour forex
                if asset_type == "forex" and 'date' in chunk_df.columns and 'time' in chunk_df.columns:
                    chunk_df['timestamp'] = (chunk_df['date'].astype(str) + ' ' + chunk_df['time'].astype(str))
                    chunk_df = chunk_df.drop(columns=['date', 'time'])
                
                processed_chunk = self._process_data_chunk(chunk_df, asset_type, symbol)
                self._add_to_buffer(processed_chunk, asset_type, symbol)
    
    def _parse_timestamp_strict(self, df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
        """
        Parse strict des timestamps et normalise en UTC tz-aware pour BRONZE.
        - forex: format "%Y%m%d %H:%M:%S"
        - autres: inference pandas
        - crypto: timestamps fournis en UTC (localisation directe UTC)
        - autres assets: localiser en Eastern puis convertir en UTC
        """
        try:
            # 1) Parsing strict
            if asset_type == "forex":
                df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y%m%d %H:%M:%S", errors="raise")
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors="raise")

            # 2) Normalisation de timezone → UTC tz-aware
            if pd.api.types.is_datetime64tz_dtype(df['timestamp']):
                # Déjà timezone-aware → convertir vers UTC si nécessaire
                df['timestamp'] = df['timestamp'].dt.tz_convert(self.utc_tz)
            else:
                # tz-naive → localiser suivant l'asset
                if asset_type == "crypto":
                    df['timestamp'] = df['timestamp'].dt.tz_localize(self.utc_tz)
                else:
                    # Par défaut (stock/etf/future/forex/index): Eastern → UTC
                    df['timestamp'] = df['timestamp'].dt.tz_localize(self.eastern_tz, ambiguous="infer")
                    df['timestamp'] = df['timestamp'].dt.tz_convert(self.utc_tz)

            return df

        except Exception as e:
            logger.error(f"Erreur parsing timestamp: {e}")
            logger.error(f"Sample timestamps: {df['timestamp'].head()}")
            logger.error(f"Type d'asset: {asset_type}")
            raise
    
    # Fonction supprimée car intégrée directement dans _parse_timestamp_strict
    
    def _validate_required_columns(self, df: pd.DataFrame, asset_type: str):
        """
        Valide les colonnes requises selon l'asset_class
        """
        required_cols = self.COLUMN_REQUIREMENTS.get(asset_type, [])
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Colonne requise manquante pour {asset_type}: {col}")
    
    def _process_data_chunk(self, df: pd.DataFrame, asset_type: str, symbol: str) -> pd.DataFrame:
        """
        Traite un chunk de données selon le type d'asset (BRONZE uniquement)
        """
        logger.info(f"=== DÉBUT _process_data_chunk ===")
        logger.info(f"Taille du chunk: {len(df)} lignes")
        logger.info(f"Colonnes initiales: {list(df.columns)}")
        
        # 1. Valider les colonnes requises
        logger.info("Validation des colonnes requises...")
        self._validate_required_columns(df, asset_type)
        logger.info("✅ Colonnes validées")
        
        # 2. Parser strict des timestamps (UTC tz-aware garanti en sortie)
        logger.info("Parsing des timestamps...")
        df = self._parse_timestamp_strict(df, asset_type)
        logger.info(f"Timestamps parsés, type: {df['timestamp'].dtype}")
        
        # 3. Ajouter les colonnes de partitionnement
        logger.info("Ajout des colonnes de partitionnement...")
        df['asset_class'] = asset_type
        df['symbol'] = symbol
        
        # Vérifier les timestamps avant extraction de l'année
        logger.info(f"Sample timestamps: {df['timestamp'].head()}")
        logger.info(f"Timestamp min: {df['timestamp'].min()}")
        logger.info(f"Timestamp max: {df['timestamp'].max()}")
        
        # S'assurer d'un dtype explicite datetime64[ns, UTC]
        try:
            df['timestamp'] = df['timestamp'].astype('datetime64[ns, UTC]')
        except Exception:
            # Fallback robuste: re-parser puis localiser UTC
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if not pd.api.types.is_datetime64tz_dtype(df['timestamp']):
                df['timestamp'] = df['timestamp'].dt.tz_localize(self.utc_tz)
            df['timestamp'] = df['timestamp'].astype('datetime64[ns, UTC]')

        # Extraire l'année en UTC (après normalisation)
        df['year'] = df['timestamp'].dt.year
        logger.info(f"Années détectées: {sorted(df['year'].unique())}")
        
        # 4. Gestion du volume
        if 'volume' not in df.columns:
            logger.info(f"Ajout de la colonne volume pour {asset_type}")
            df['volume'] = 0 if asset_type == "index" else None
        elif asset_type == "index":
            logger.info("Force volume=0 pour index")
            df['volume'] = 0
        
        logger.info(f"Colonnes finales: {list(df.columns)}")
        logger.info(f"=== FIN _process_data_chunk ===")
        return df
    
    def _add_to_buffer(self, df: pd.DataFrame, asset_type: str, symbol: str):
        """
        Ajoute un chunk au buffer pour écriture groupée
        """
        partition_key = (asset_type, symbol)
        logger.info(f"Ajout au buffer pour la partition: {partition_key}")
        
        if partition_key not in self.chunk_buffer:
            self.chunk_buffer[partition_key] = []
            logger.info(f"Nouveau buffer créé pour {partition_key}")
        
        self.chunk_buffer[partition_key].append(df)
        logger.info(f"Chunk ajouté au buffer {partition_key}, total chunks: {len(self.chunk_buffer[partition_key])}")
        
        # Vérifier si le buffer doit être vidé
        total_rows = sum(len(chunk) for chunk in self.chunk_buffer[partition_key])
        logger.info(f"Total lignes dans le buffer {partition_key}: {total_rows}")
        
        if total_rows >= self.buffer_size_limit:
            logger.info(f"Buffer {partition_key} plein ({total_rows} lignes), vidage...")
            self._flush_buffer(partition_key)
        else:
            logger.info(f"Buffer {partition_key} pas encore plein ({total_rows}/{self.buffer_size_limit})")
    
    def _flush_buffer(self, partition_key: Tuple[str, str]):
        """
        Vide un buffer spécifique vers Parquet
        """
        if partition_key not in self.chunk_buffer or not self.chunk_buffer[partition_key]:
            logger.warning(f"Buffer {partition_key} vide ou inexistant")
            return
        
        asset_type, symbol = partition_key
        chunks = self.chunk_buffer[partition_key]
        
        logger.info(f"Vidage du buffer {partition_key}: {len(chunks)} chunks")
        
        # Concaténer tous les chunks
        combined_df = pd.concat(chunks, ignore_index=True)
        logger.info(f"DataFrame combiné: {len(combined_df)} lignes, colonnes: {list(combined_df.columns)}")
        
        # Écrire en Parquet avec partitionnement
        self._write_partition_to_parquet(combined_df, asset_type, symbol)
        
        # Vider le buffer
        self.chunk_buffer[partition_key] = []
        logger.info(f"Buffer {partition_key} vidé")
    
    def _flush_all_buffers(self):
        """
        Vide tous les buffers restants
        """
        for partition_key in list(self.chunk_buffer.keys()):
            self._flush_buffer(partition_key)
    
    def _write_partition_to_parquet(self, df: pd.DataFrame, asset_type: str, symbol: str):
        """
        Écrit une partition en Parquet avec le bon chemin
        """
        logger.info(f"=== DÉBUT _write_partition_to_parquet ===")
        logger.info(f"Asset type: {asset_type}")
        logger.info(f"Symbole: {symbol}")
        logger.info(f"Taille DataFrame: {len(df)} lignes")
        
        if df.empty:
            logger.warning("DataFrame vide, pas d'écriture")
            return
        
        # Chemin BRONZE : <base>/bronze/asset_class=<ASSET>/symbol=<SYMBOL>/
        base_dir = (self.bronze_dir / 
                   f"asset_class={asset_type}" / 
                   f"symbol={symbol}")
        
        logger.info(f"Répertoire de base: {base_dir}")
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Normaliser le type avant conversion Arrow
        if not pd.api.types.is_datetime64tz_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(self.utc_tz)
        df['timestamp'] = df['timestamp'].astype('datetime64[ns, UTC]')

        # Convertir en table PyArrow
        table = pa.Table.from_pandas(df)
        logger.info(f"Table PyArrow créée: {table.num_rows} lignes, {table.num_columns} colonnes")
        
        # Écrire avec compression zstd et partitionnement par année
        logger.info("Écriture en Parquet avec compression zstd...")
        pq.write_to_dataset(
            table,
            base_dir,
            partition_cols=['year'],
            compression="zstd"
        )
            
        logger.info(f"✅ Données écrites avec succès dans: {base_dir}")
        logger.info(f"=== FIN _write_partition_to_parquet ===")
    
    def process_csv_file(self, file_path: str, asset_type: str, symbol: str) -> Dict[str, int]:
        """
        Traite un fichier CSV/TXT directement
        """
        logger.info(f"=== DÉBUT process_csv_file ===")
        logger.info(f"Fichier: {file_path}")
        logger.info(f"Asset type: {asset_type}")
        logger.info(f"Symbole: {symbol}")
        
        if not os.path.exists(file_path):
            logger.error(f"Fichier non trouvé: {file_path}")
            raise FileNotFoundError(f"Fichier non trouvé : {file_path}")
        
        results = {"fichiers_traités": 0, "lignes_traitées": 0, "erreurs": 0}
        
        try:
            # Traiter le fichier en streaming
            # Utiliser les colonnes standard
            
            chunk_size = 500000  # 500K lignes par chunk
            logger.info(f"Taille des chunks: {chunk_size}")
            
            chunk_count = 0
            # Choix des colonnes selon l'asset
            if asset_type == "forex":
                read_names = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
            else:
                read_names = self.STANDARD_COLUMNS
            
            # Lire le fichier avec les bons paramètres
            for chunk_num, chunk_df in enumerate(pd.read_csv(
                file_path,
                chunksize=chunk_size,
                names=read_names,
                header=None,
                sep=','
            )):
                try:
                    chunk_count += 1
                    logger.info(f"Traitement du chunk {chunk_num + 1}, taille: {len(chunk_df)} lignes")
                    
                    # Combiner date+time pour forex
                    if asset_type == "forex" and 'date' in chunk_df.columns and 'time' in chunk_df.columns:
                        chunk_df['timestamp'] = (chunk_df['date'].astype(str) + ' ' + chunk_df['time'].astype(str))
                        chunk_df = chunk_df.drop(columns=['date', 'time'])
                    
                    # Traiter le chunk
                    processed_chunk = self._process_data_chunk(chunk_df, asset_type, symbol)
                    logger.info(f"Chunk {chunk_num + 1} traité, colonnes: {list(processed_chunk.columns)}")
                    
                    # Ajouter au buffer
                    self._add_to_buffer(processed_chunk, asset_type, symbol)
                    logger.info(f"Chunk {chunk_num + 1} ajouté au buffer")
                    
                    results["fichiers_traités"] = 1
                    results["lignes_traitées"] += len(processed_chunk)
                    
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du chunk {chunk_num}: {e}")
                    results["erreurs"] += 1
                    
            logger.info(f"Tous les chunks traités: {chunk_count}")
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du fichier {file_path}: {e}")
            raise
        
        # Écrire les buffers restants
        logger.info("Vidage des buffers restants...")
        self._flush_all_buffers()
        
        logger.info(f"=== FIN process_csv_file - Résultats: {results} ===")
        return results
    
    def read_symbol_data(self, asset_type: str, symbol: str, start_year: Optional[int] = None, end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Lit les données d'un symbole avec filtrage optionnel par année
        
        Args:
            asset_type: Type d'asset (etf, stock, etc.)
            symbol: Symbole à lire
            start_year: Année de début (optionnel)
            end_year: Année de fin (optionnel)
            
        Returns:
            DataFrame avec les données triées par timestamp
        """
        # Construire le chemin
        symbol_dir = (self.bronze_dir / 
                     f"asset_class={asset_type}" / 
                     f"symbol={symbol}")
        
        if not symbol_dir.exists():
            raise FileNotFoundError(f"Données non trouvées pour {symbol} ({asset_type})")
            
        # Construire le filtre sur les années
        filters = None
        if start_year is not None or end_year is not None:
            filters = []
            if start_year is not None:
                filters.append(('year', '>=', start_year))
            if end_year is not None:
                filters.append(('year', '<=', end_year))
        
        # Lire le dataset
        dataset = ds.dataset(symbol_dir, format="parquet")
        table = dataset.to_table(filter=filters)
        df = table.to_pandas()
        
        # Trier par timestamp
        if not df.empty:
            df = df.sort_values('timestamp')
            
        return df
    
    def get_storage_info(self) -> Dict[str, Dict]:
        """
        Retourne les informations de stockage BRONZE
        """
        info = {}
        
        for asset_type in self.asset_types:
            asset_dir = self.bronze_dir / f"asset_class={asset_type}"
            if asset_dir.exists():
                # Compter les symboles
                symbols = [d.name.replace('symbol=', '') for d in asset_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('symbol=')]
                
                # Compter les années totales
                total_years = 0
                total_files = 0
                total_size = 0
                
                for symbol_dir in asset_dir.iterdir():
                    if symbol_dir.name.startswith('symbol='):
                        for year_dir in symbol_dir.iterdir():
                            if year_dir.name.startswith('year='):
                                total_years += 1
                                year_files = list(year_dir.glob('*.parquet'))
                                total_files += len(year_files)
                                total_size += sum(f.stat().st_size for f in year_files)
                
                info[asset_type] = {
                    "nombre_sous_jacents": len(symbols),
                    "sous_jacents": symbols,
                    "années_total": total_years,
                    "nombre_fichiers": total_files,
                    "taille_totale_gb": round(total_size / (1024**3), 2),
                    "chemin": str(asset_dir)
                }
            else:
                info[asset_type] = {
                    "nombre_sous_jacents": 0,
                    "sous_jacents": [],
                    "années_total": 0,
                    "nombre_fichiers": 0,
                    "taille_totale_gb": 0,
                    "chemin": str(asset_dir)
                }
        
        return info

    def _detect_symbol_from_path(self, file_path: str, asset_type: str) -> str:
        """
        Détecte automatiquement le symbole depuis le nom du fichier
        """
        path = Path(file_path)
        file_name = path.stem  # Nom sans extension
        
        # Enlever le préfixe temp_ s'il existe
        if file_name.startswith('temp_'):
            file_name = file_name[5:]
        
        # Prendre le symbole avant le premier underscore
        symbol = file_name.split('_')[0].upper()
        
        logger.info(f"Symbole détecté : {symbol}")
        return symbol
