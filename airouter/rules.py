from pathlib import Path

HEADER = "<!-- Copied from ~/.claude/CLAUDE.md by `ai`. Edit that file instead; this copy is overwritten. -->\n\n"


def source() -> Path:
    return Path.home() / ".claude" / "CLAUDE.md"


def targets() -> list[Path]:
    home = Path.home()
    return [home / ".gemini" / "GEMINI.md", home / ".copilot" / "copilot-instructions.md"]


def sync() -> list[Path]:
    src = source()
    if not src.exists():
        return []
    text = HEADER + src.read_text(encoding="utf-8-sig")
    changed = []
    for target in targets():
        if target.exists() and target.read_text(encoding="utf-8-sig") == text:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        changed.append(target)
    return changed
