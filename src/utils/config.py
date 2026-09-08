from __future__ import annotations
import hashlib
import json 
from pathlib import Path
from typing import Any 
import yaml

class Config(dict):

    def __init__(self, data: dict):
        super().__init__(data)
        for key, value in data.items():
            if isinstance(value,dict):
                self[key]=Config(value)

    def __getattr__(self, name:str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"No config key '{name}'. Available: {sorted(self.keys())}"

            ) from exc

    def __setattr__(self, name:str, value:Any) -> None:
        self[name]=value

def load_config(path:str |Path) -> Config:
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return Config(yaml.safe_load(handle))

def config_fingerprint(cfg:Config, sections: tuple[str, ...]) -> str:
    subset= {section: cfg.get(section,{}) for section in sections}
    blob = json.dumps(subset,sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
        