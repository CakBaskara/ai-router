import math
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Candidate:
    provider: str
    recent_successes: int
    attempts: int
    success_rate: float | None
    reliability_lower_bound: float | None
    latency_p95: float | None
    input_tokens: int
    output_tokens: int
    quota_headroom: int | None
    weekly_headroom: int | None
    quota_fresh: bool
    below_reserve: bool
    unreliable: bool
    cost_verified: bool = False

    def log_data(self) -> dict:
        return asdict(self)


def _wilson_lower(successes: int, total: int, z: float = 1.96) -> float | None:
    if not total:
        return None
    rate = successes / total
    denominator = 1 + z * z / total
    center = rate + z * z / (2 * total)
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    return round((center - margin) / denominator, 4)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 2)


def _headroom(telemetry: dict, window: str, fresh: bool) -> int | None:
    used = (telemetry.get(window) or {}).get("used")
    return max(0, round(100 - used)) if fresh and isinstance(used, (int, float)) else None


def candidates(cfg: dict, usage: dict | None = None, quota: dict | None = None,
               history: list[dict] | None = None, tier: str | None = None,
               now: float | None = None) -> list[Candidate]:
    usage = usage or {}
    quota = quota or {}
    history = history or []
    now = now or time.time()
    settings = cfg.get("optimizer", {})
    reserve = int(settings.get("quota_reserve", 15))
    weekly_reserve = int(settings.get("weekly_quota_reserve", 10))
    max_age = int(settings.get("quota_max_age", 180))
    minimum = int(settings.get("reliability_min_samples", 5))
    floor = float(settings.get("reliability_floor", 0.6))
    ranked = []
    for provider in cfg["providers"]:
        telemetry = quota.get(provider) or {}
        observed = telemetry.get("at")
        fresh = isinstance(observed, (int, float)) and 0 <= now - observed <= max_age
        five_hour = _headroom(telemetry, "5h", fresh)
        weekly = _headroom(telemetry, "week", fresh)
        rows = [row for row in history if row.get("provider") == provider
                and (not tier or row.get("tier_effective", row.get("tier")) == tier)]
        attempts = len(rows)
        successes = sum(row.get("exit") == 0 and not row.get("error") for row in rows)
        rate = round(successes / attempts, 4) if attempts else None
        lower = _wilson_lower(successes, attempts) if attempts >= minimum else None
        latencies = [float(row["seconds"]) for row in rows
                     if isinstance(row.get("seconds"), (int, float)) and row["seconds"] >= 0]
        actual = [row.get("actual_usage") or {} for row in rows]
        input_tokens = sum(int(row.get("input_tokens") or 0) for row in actual)
        output_tokens = sum(int(row.get("output_tokens") or 0) for row in actual)
        below = ((five_hour is not None and five_hour < reserve)
                 or (weekly is not None and weekly < weekly_reserve))
        unreliable = attempts >= minimum and rate is not None and rate < floor
        ranked.append(Candidate(
            provider=provider,
            recent_successes=int(usage.get(provider, 0)),
            attempts=attempts,
            success_rate=rate,
            reliability_lower_bound=lower,
            latency_p95=_percentile(latencies, 0.95),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            quota_headroom=five_hour,
            weekly_headroom=weekly,
            quota_fresh=fresh,
            below_reserve=below,
            unreliable=unreliable,
        ))
    return ranked


def rank_candidates(cfg: dict, sticky: str | None = None, usage: dict | None = None,
                    quota: dict | None = None, history: list[dict] | None = None,
                    tier: str | None = None, now: float | None = None) -> list[Candidate]:
    listed = candidates(cfg, usage, quota, history, tier, now)
    order = {provider: index for index, provider in enumerate(cfg["providers"])}
    listed.sort(key=lambda item: (
        item.below_reserve,
        item.unreliable,
        -(item.reliability_lower_bound if item.reliability_lower_bound is not None else -1),
        item.recent_successes,
        item.latency_p95 if item.latency_p95 is not None else float("inf"),
        order[item.provider],
    ))
    selected = next((item for item in listed if item.provider == sticky), None)
    if selected and not selected.below_reserve and not selected.unreliable:
        listed.remove(selected)
        listed.insert(0, selected)
    return listed


def rank_providers(cfg: dict, sticky: str | None = None, usage: dict | None = None,
                   quota: dict | None = None, history: list[dict] | None = None,
                   tier: str | None = None, now: float | None = None) -> list[str]:
    return [item.provider for item in rank_candidates(cfg, sticky, usage, quota, history, tier, now)]
