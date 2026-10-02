import json
import sys
from collections import Counter
from datetime import date

from . import config


def say(msg: str):
    print(f"\033[2m{msg}\033[0m", file=sys.stderr, flush=True)


def log(entry: dict):
    path = config.log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def usage_today() -> Counter:
    path = config.log_path()
    today = date.today().isoformat()
    counts = Counter()
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith('{"ts": "' + today):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("exit") == 0 and entry.get("provider"):
            counts[entry["provider"]] += 1
    return counts
