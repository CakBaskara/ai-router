from airouter import rules


def test_sync_copies_claude_rules_to_other_agents(tmp_path, monkeypatch):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "CLAUDE.md").write_text("# rules\n- talk in Bahasa Indonesia\n", encoding="utf-8")
    changed = rules.sync()
    assert len(changed) == 2
    for target in rules.targets():
        text = target.read_text(encoding="utf-8")
        assert text.startswith(rules.HEADER) and "Bahasa Indonesia" in text
    assert rules.sync() == []


def test_sync_without_source_does_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    assert rules.sync() == []
