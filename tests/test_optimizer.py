from airouter.optimizer import candidates, rank_providers


CFG = {"providers": ["claude", "codex"], "optimizer": {"quota_reserve": 15, "quota_max_age": 180}}


def test_missing_quota_uses_the_existing_success_count_fallback():
    assert rank_providers(CFG, usage={"claude": 5, "codex": 1}) == ["codex", "claude"]
    rows = candidates(CFG, usage={"claude": 5})
    assert all(not row.cost_verified and row.quota_headroom is None for row in rows)


def test_fresh_quota_below_reserve_moves_provider_to_the_end():
    now = 1_000
    quota = {
        "claude": {"at": now, "5h": {"used": 91}},
        "codex": {"at": now, "5h": {"used": 20}},
    }
    assert rank_providers(CFG, sticky="claude", quota=quota, now=now) == ["codex", "claude"]


def test_stale_quota_does_not_claim_a_provider_is_depleted():
    quota = {"claude": {"at": 1, "5h": {"used": 99}}}
    assert rank_providers(CFG, sticky="claude", quota=quota, now=1_000) == ["claude", "codex"]


def test_sticky_provider_stays_first_when_it_has_reserve():
    now = 1_000
    quota = {"claude": {"at": now, "5h": {"used": 10}}}
    assert rank_providers(CFG, sticky="claude", usage={"claude": 9}, quota=quota, now=now)[0] == "claude"


def test_weekly_reserve_is_a_separate_hard_limit():
    now = 1_000
    quota = {
        "claude": {"at": now, "5h": {"used": 10}, "week": {"used": 95}},
        "codex": {"at": now, "5h": {"used": 50}, "week": {"used": 20}},
    }
    assert rank_providers(CFG, sticky="claude", quota=quota, now=now) == ["codex", "claude"]


def test_reliability_latency_and_tokens_come_from_matching_tier_history():
    history = [
        {"provider": "claude", "tier_effective": "heavy", "exit": 0, "seconds": 4,
         "actual_usage": {"input_tokens": 10, "output_tokens": 2}},
        {"provider": "claude", "tier_effective": "heavy", "exit": 1, "seconds": 8,
         "actual_usage": {"input_tokens": 5, "output_tokens": 1}},
        {"provider": "claude", "tier_effective": "light", "exit": 0, "seconds": 1},
    ]
    row = next(item for item in candidates(CFG, history=history, tier="heavy") if item.provider == "claude")
    assert (row.attempts, row.success_rate, row.latency_p95) == (2, 0.5, 8)
    assert (row.input_tokens, row.output_tokens) == (15, 3)


def test_provider_below_reliability_floor_loses_sticky_priority():
    cfg = {**CFG, "optimizer": {**CFG["optimizer"], "reliability_min_samples": 3,
                                "reliability_floor": 0.6}}
    history = ([{"provider": "claude", "tier": "heavy", "exit": 1}] * 3
               + [{"provider": "codex", "tier": "heavy", "exit": 0}] * 3)
    assert rank_providers(cfg, sticky="claude", history=history, tier="heavy")[0] == "codex"


def test_small_sample_does_not_gain_reliability_priority():
    cfg = {**CFG, "optimizer": {**CFG["optimizer"], "reliability_min_samples": 5}}
    history = [{"provider": "claude", "tier": "heavy", "exit": 0}]
    rows = candidates(cfg, history=history, tier="heavy")
    claude = next(row for row in rows if row.provider == "claude")
    assert claude.success_rate == 1 and claude.reliability_lower_bound is None
    assert rank_providers(cfg, usage={"claude": 1}, history=history, tier="heavy")[0] == "codex"
