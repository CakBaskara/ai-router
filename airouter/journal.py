import json
import sys

from . import config


def say(msg: str):
    print(f"\033[2m{msg}\033[0m", file=sys.stderr, flush=True)


def log(entry: dict):
    path = config.log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
