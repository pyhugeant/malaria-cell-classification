import copy
from dataclasses import dataclass
from typing import Any, Dict
import yaml

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def deep_get(d: Dict[str, Any], keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def deep_copy(d: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(d)

def resolve_device(device_cfg: str) -> str:
    import torch
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg
