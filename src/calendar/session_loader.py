import json
from pathlib import Path
from typing import Dict, Optional

import pytz


class TradingSessionTemplates:
    def __init__(self, config_path: str = "config/trading_sessions.json") -> None:
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Sessions config not found: {self.config_path}")
        self.templates: Dict[str, dict] = json.loads(self.config_path.read_text(encoding="utf-8"))

    def get(self, name: str) -> dict:
        tpl = self.templates.get(name)
        if not tpl:
            raise KeyError(f"Unknown session template: {name}")
        # validate timezone
        pytz.timezone(tpl["timezone"])  # raises if invalid
        return tpl


class SymbolSessionRegistry:
    """
    Registry that maps a symbol (or asset_class=* wildcard) to a session template name.
    Persistent file optional (JSON).
    """

    def __init__(self, registry_path: str = "config/symbol_sessions.json") -> None:
        self.path = Path(registry_path)
        if self.path.exists():
            self.map: Dict[str, str] = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.map = {}

    def set(self, key: str, template_name: str) -> None:
        self.map[key.upper()] = template_name
        self._save()

    def get(self, key: str) -> Optional[str]:
        return self.map.get(key.upper())

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.map, indent=2), encoding="utf-8")
    
    def auto_map_by_directory(self, base_dir: Path, template_mapping: dict) -> dict:
        """
        Mapping automatique par répertoire.
        template_mapping: {"stock": "equity_us_rth", "future": "futures_grains_cme", ...}
        Retourne: {"mapped": count, "total": count, "details": list}
        """
        mapped_count = 0
        total_count = 0
        details = []
        
        for asset_class, template in template_mapping.items():
            # Chercher dans Bronze et Silver
            for tier in ["bronze", "silver", "daily"]:
                asset_dir = base_dir / tier / f"asset_class={asset_class}"
                if not asset_dir.exists():
                    continue
                    
                for symbol_dir in asset_dir.iterdir():
                    if symbol_dir.is_dir() and symbol_dir.name.startswith("symbol="):
                        symbol = symbol_dir.name.replace("symbol=", "")
                        total_count += 1
                        
                        # Si pas déjà mappé, appliquer le template
                        if symbol.upper() not in self.map:
                            self.set(symbol, template)
                            mapped_count += 1
                            details.append(f"{symbol} → {template}")
        
        return {
            "mapped": mapped_count,
            "total": total_count, 
            "details": details
        }
    
    def search_symbols(self, query: str, limit: int = 50) -> list:
        """
        Recherche de symboles par pattern (case-insensitive).
        Retourne max `limit` résultats.
        """
        if not query:
            return []
            
        query_lower = query.lower()
        matches = []
        
        for symbol, template in self.map.items():
            if query_lower in symbol.lower():
                matches.append({"symbol": symbol, "template": template})
                if len(matches) >= limit:
                    break
        
        return sorted(matches, key=lambda x: x["symbol"])
    
    def get_stats(self) -> dict:
        """Statistiques du registry."""
        from collections import Counter
        templates = Counter(self.map.values())
        return {
            "total_symbols": len(self.map),
            "templates_used": dict(templates),
        }
    
    def get_all(self) -> dict:
        """Retourne tous les mappings (pour compatibilité)."""
        return self.map.copy()
