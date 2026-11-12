from airouter.failover import (AttemptResult, Phase, auto_retry_allowed, changed_files, classify_attempt,
                               workspace_snapshot)


def test_retry_is_allowed_only_before_dispatch_without_effects():
    safe = AttemptResult(1, Phase.NOT_DISPATCHED, external_effects_possible=False)
    assert auto_retry_allowed(safe)
    assert not auto_retry_allowed(AttemptResult(1, Phase.UNKNOWN))
    assert not auto_retry_allowed(AttemptResult(130, Phase.NOT_DISPATCHED, cancelled=True,
                                                external_effects_possible=False))
    assert not auto_retry_allowed(AttemptResult(1, Phase.EFFECTS_OBSERVED, file_changes_detected=True))


def test_tree_snapshot_detects_content_and_untracked_changes(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("before", encoding="utf-8")
    before = workspace_snapshot(tmp_path)
    source.write_text("after!", encoding="utf-8")
    added = tmp_path / "new.txt"
    added.write_text("new", encoding="utf-8")
    after = workspace_snapshot(tmp_path)
    assert changed_files(before, after) == ("new.txt", "source.txt")


def test_unchanged_preexisting_files_do_not_count_as_attempt_effects(tmp_path):
    (tmp_path / "already-dirty.txt").write_text("user data", encoding="utf-8")
    before = workspace_snapshot(tmp_path)
    after = workspace_snapshot(tmp_path)
    result = classify_attempt(1, True, False, before, after)
    assert result.phase is Phase.UNKNOWN
    assert not result.file_changes_detected and result.changed_files == ()
    assert not auto_retry_allowed(result)


def test_change_after_dispatch_is_reported_as_observed_effect(tmp_path):
    before = workspace_snapshot(tmp_path)
    (tmp_path / "generated.txt").write_text("result", encoding="utf-8")
    after = workspace_snapshot(tmp_path)
    result = classify_attempt(1, True, False, before, after)
    assert result.phase is Phase.EFFECTS_OBSERVED
    assert result.changed_files == ("generated.txt",)


def test_spawn_failure_without_workspace_change_is_safe_to_retry(tmp_path):
    before = workspace_snapshot(tmp_path)
    after = workspace_snapshot(tmp_path)
    result = classify_attempt(1, False, False, before, after)
    assert result.phase is Phase.NOT_DISPATCHED and auto_retry_allowed(result)
