import json, yaml
from pathlib import Path

def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def yolo_class_names(data_yaml_path):
    cfg = load_yaml(data_yaml_path)
    names = cfg.get("names", {})
    if isinstance(names, dict):
        return [names[k] for k in sorted(map(int, names.keys()))]
    return list(names)
