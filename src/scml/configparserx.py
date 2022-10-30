from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path
from typing import Dict

__all__ = ["as_dict", "load"]


def as_dict(conf: ConfigParser) -> Dict:
    return {s: dict(conf.items(s)) for s in conf.sections()}


def load(filepath: Path) -> ConfigParser:
    if not filepath.is_file():
        raise ValueError(f"File does not exist: {str(filepath)}")
    conf = ConfigParser(interpolation=ExtendedInterpolation())
    conf.read(str(filepath), encoding="utf-8")
    conf["DEFAULT"]["conf_dir"] = str(filepath.resolve().parent)
    return conf
