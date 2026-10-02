import os
import tomllib
from pathlib import Path

PACKAGE_CONFIG = Path(__file__).with_name("config.toml")
REPO_ROOT = Path(__file__).resolve().parent.parent


def load() -> dict:
    path = Path(os.environ.get("AI_ROUTER_CONFIG", PACKAGE_CONFIG))
    return tomllib.loads(path.read_text(encoding="utf-8-sig"))


def log_path() -> Path:
    return Path(os.environ.get("AI_ROUTER_LOG", REPO_ROOT / "logs" / "routes.jsonl"))
