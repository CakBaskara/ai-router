import re
from dataclasses import dataclass


TIERS = ("light", "medium", "heavy")
CONTINUITY = re.compile(
    r"\b(?:lanjut(?:kan)?|continue|tadi|sebelumnya|masih sama|error(?:nya|-nya)?|itu|tersebut|"
    r"yang sama|hasilnya|kodenya|testnya|tesnya|fix it|same error|that|previous)\b",
    re.IGNORECASE,
)
NEW_TOPIC = re.compile(
    r"^(?:apa itu|what is|what's|arti |terjemah|translate|cara |how to|berapa|who is|siapa )",
    re.IGNORECASE,
)
WORDS = re.compile(r"[a-zA-Z0-9_]{3,}")
STOPWORDS = {
    "apa", "itu", "yang", "dan", "atau", "untuk", "dari", "dengan", "coba", "tolong", "bisa",
    "what", "this", "that", "the", "and", "for", "from", "with", "please", "can", "you",
}


@dataclass(frozen=True)
class RoutingDecision:
    turn_tier: str
    context_tier: str
    effective_tier: str
    related_to_previous: bool
    reason: str


def _higher(first: str, second: str) -> str:
    return first if TIERS.index(first) >= TIERS.index(second) else second


def _words(text: str) -> set[str]:
    return {word.lower() for word in WORDS.findall(text) if word.lower() not in STOPWORDS}


def related_to_previous(message: str, previous_message: str | None, ambiguous: bool = False) -> tuple[bool, str]:
    if not previous_message:
        return False, "tidak ada konteks sebelumnya"
    normalized = " ".join(message.split())
    if NEW_TOPIC.search(normalized):
        return False, "pertanyaan mandiri baru"
    if CONTINUITY.search(normalized):
        return True, "merujuk percakapan sebelumnya"
    current, previous = _words(normalized), _words(previous_message)
    overlap = current & previous
    if len(overlap) >= 2 or (overlap and len(current) <= 5):
        return True, "topik masih sama"
    if ambiguous and len(normalized.split()) <= 5:
        return True, "tindak lanjut singkat"
    return False, "topik baru"


def decide(turn_tier: str, previous_tier: str | None, message: str, previous_message: str | None,
           ambiguous: bool = False, minimum: str = "light", pinned: str | None = None) -> RoutingDecision:
    related, reason = related_to_previous(message, previous_message, ambiguous)
    context = previous_tier if related and previous_tier in TIERS else minimum
    effective = pinned or (_higher(turn_tier, context) if related else turn_tier)
    if minimum in TIERS:
        effective = _higher(effective, minimum)
    if pinned:
        reason = f"dikunci {pinned}"
    elif effective != turn_tier and related:
        reason = f"{reason}; pertahankan konteks {context}"
    elif effective != turn_tier:
        reason = f"minimum chat {minimum}"
    return RoutingDecision(turn_tier, context, effective, related, reason)
