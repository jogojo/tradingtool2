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
