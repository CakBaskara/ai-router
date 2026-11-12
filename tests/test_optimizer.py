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
