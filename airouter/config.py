import json
import os
import sys
import tomllib
from collections import Counter
from datetime import date
from pathlib import Path

PACKAGE_CONFIG = Path(__file__).with_name("config.toml")
REPO_ROOT = Path(__file__).resolve().parent.parent


def load() -> dict:
    path = Path(os.environ.get("AI_ROUTER_CONFIG", PACKAGE_CONFIG))
    return tomllib.loads(path.read_text(encoding="utf-8-sig"))


def log_path() -> Path:
    return Path(os.environ.get("AI_ROUTER_LOG", REPO_ROOT / "logs" / "routes.jsonl"))


def say(msg: str):
    print(f"\033[2m{msg}\033[0m", file=sys.stderr, flush=True)


def log(entry: dict):
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def usage_today() -> Counter:
    path = log_path()
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


RULES_HEADER = "<!-- Copied from ~/.claude/CLAUDE.md by `ai`. Edit that file instead; this copy is overwritten. -->\n\n"


def rules_source() -> Path:
    return Path.home() / ".claude" / "CLAUDE.md"


def rules_targets() -> list[Path]:
    home = Path.home()
    return [home / ".gemini" / "GEMINI.md", home / ".copilot" / "copilot-instructions.md"]


def sync_rules() -> list[Path]:
    src = rules_source()
    if not src.exists():
        return []
    text = RULES_HEADER + src.read_text(encoding="utf-8-sig")
    changed = []
    for target in rules_targets():
        if target.exists() and target.read_text(encoding="utf-8-sig") == text:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        changed.append(target)
    return changed
