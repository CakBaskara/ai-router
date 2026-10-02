import re
from dataclasses import dataclass, field

TIERS = ("light", "medium", "heavy")


@dataclass
class Verdict:
    tier: str
    score: int
    reasons: list[str] = field(default_factory=list)
    ambiguous: bool = False


def _hits(text: str, words: list[str]) -> list[str]:
    return [w for w in words if re.search(r"(?<!\w)" + re.escape(w.lower()), text)]


def _looks_like_code(prompt: str) -> bool:
    if "```" in prompt or "Traceback" in prompt:
        return True
    if re.search(r"^\s+at \S+", prompt, re.M):
        return True
    return prompt.count("\n") >= 10


def classify(prompt: str, rules: dict, repo: bool = False) -> Verdict:
    text = prompt.lower()
    score = 0
    reasons = []

    words = len(prompt.split())
    if words >= 150:
        score += 2
        reasons.append(f"{words} words")
    elif words >= 50:
        score += 1
        reasons.append(f"{words} words")

    if _looks_like_code(prompt):
        score += 1
        reasons.append("code or log")

    heavy = _hits(text, rules.get("heavy", []))
    medium = _hits(text, rules.get("medium", []))
    light = _hits(text, rules.get("light", []))
    if heavy:
        score += 3
        reasons.append("heavy: " + ", ".join(heavy))
    if medium:
        score += 1
        reasons.append("medium: " + ", ".join(medium))
    if light:
        score -= 1
        reasons.append("light: " + ", ".join(light))

    if repo:
        score += 1
        reasons.append("repo session")

    tier = "light" if score <= 0 else "medium" if score <= 2 else "heavy"
    ambiguous = not (heavy or medium or light)
    return Verdict(tier, score, reasons, ambiguous)
