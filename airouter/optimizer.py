import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    provider: str
    recent_successes: int
    quota_headroom: int | None
    quota_fresh: bool
    below_reserve: bool
    cost_verified: bool = False


def candidates(cfg: dict, usage: dict | None = None, quota: dict | None = None,
               now: float | None = None) -> list[Candidate]:
    usage = usage or {}
    quota = quota or {}
    now = now or time.time()
    settings = cfg.get("optimizer", {})
    reserve = int(settings.get("quota_reserve", 15))
    max_age = int(settings.get("quota_max_age", 180))
    ranked = []
    for provider in cfg["providers"]:
        telemetry = quota.get(provider) or {}
        observed = telemetry.get("at")
        fresh = isinstance(observed, (int, float)) and 0 <= now - observed <= max_age
        window = telemetry.get("5h") or {}
        used = window.get("used")
        headroom = max(0, round(100 - used)) if fresh and isinstance(used, (int, float)) else None
        ranked.append(Candidate(provider, int(usage.get(provider, 0)), headroom, fresh,
                                headroom is not None and headroom < reserve))
    return ranked


def rank_providers(cfg: dict, sticky: str | None = None, usage: dict | None = None,
                   quota: dict | None = None, now: float | None = None) -> list[str]:
    listed = candidates(cfg, usage, quota, now)
    order = {provider: index for index, provider in enumerate(cfg["providers"])}
    listed.sort(key=lambda item: (item.below_reserve, item.recent_successes, order[item.provider]))
    providers = [item.provider for item in listed]
    selected = next((item for item in listed if item.provider == sticky), None)
    if selected and not selected.below_reserve:
        providers.remove(sticky)
        providers.insert(0, sticky)
    return providers
