import pytest

from airouter import attach, cli, config, learn, paths, tui

PLATFORM_CONFIG_DIR = paths.config_dir


@pytest.fixture
def storage(tmp_path, monkeypatch):
    for name in ("AI_ROUTER_CONFIG", "AI_ROUTER_LOG", "AI_ROUTER_LABELS", "AI_ROUTER_ATTACH", "AI_ROUTER_MODELS"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def test_paths_use_platform_directories_without_creating_them(storage, monkeypatch):
    monkeypatch.delenv("AI_ROUTER_DATA_DIR", raising=False)
    seen = []

    def user_data(appname, **options):
        seen.append((appname, options))
        return str(storage / "local")

    monkeypatch.setattr(paths, "user_data_dir", user_data)
    assert config.log_path() == storage / "local" / "logs" / "routes.jsonl"
    assert learn.labels_path() == storage / "local" / "logs" / "labels.jsonl"
    assert learn.model_folder() == storage / "local" / "models" / learn.MODEL_REPO.split("/")[1]
    assert attach.folder() == storage / "local" / "attachments"
    assert all(name == "AI-Router" and opts == {"appauthor": False} for name, opts in seen)
    assert not (storage / "local").exists()
    assert not paths.config_dir().exists()


def test_windows_config_uses_roaming(storage, monkeypatch):
    calls = []

    def user_config(appname, **options):
        calls.append((appname, options))
        return str(storage / "roaming" / appname)

    monkeypatch.setattr(paths, "user_config_dir", user_config)
    assert PLATFORM_CONFIG_DIR() == storage / "roaming" / "AI-Router"
    assert calls == [("AI-Router", {"appauthor": False, "roaming": True})]


def test_data_override_and_derived_files(storage):
    root = storage / "data"
    assert config.log_path() == root / "logs" / "routes.jsonl"
    assert config.quota_path() == root / "logs" / "quota.json"
    assert learn.labels_path() == root / "logs" / "labels.jsonl"
    assert learn.vectors_path() == root / "logs" / "ml_vectors.npz"
    assert learn.settings_path() == root / "logs" / "ml_settings.json"
    assert learn.audit_path() == root / "logs" / "ml_audit.jsonl"
    assert learn.teach_lock() == root / "logs" / "ml_teach.lock"
    assert learn.replies_path() == root / "logs" / "replies.jsonl"
    assert learn.style_path() == root / "logs" / "style.md"
    assert learn.style_state_path() == root / "logs" / "style_state.json"
    assert attach.folder() == root / "attachments"
    assert learn.model_folder().parent == root / "models"
    assert not root.exists()


def test_individual_overrides_keep_existing_sibling_rules(storage, monkeypatch):
    for key, value in {"AI_ROUTER_LOG": "route/custom.jsonl", "AI_ROUTER_LABELS": "learn/custom.jsonl",
                       "AI_ROUTER_ATTACH": "pictures", "AI_ROUTER_MODELS": "weights"}.items():
        monkeypatch.setenv(key, str(storage / value))
    assert config.log_path() == storage / "route" / "custom.jsonl"
    assert config.quota_path() == storage / "route" / "quota.json"
    assert learn.labels_path() == storage / "learn" / "custom.jsonl"
    assert learn.settings_path() == storage / "learn" / "ml_settings.json"
    assert learn.style_path() == storage / "route" / "style.md"
    assert attach.folder() == storage / "pictures"
    assert learn.model_folder().parent == storage / "weights"


def test_config_precedence_merges_defaults_and_does_not_write(storage, monkeypatch):
    assert config.config_path() == config.PACKAGE_CONFIG
    assert config.load()["chat"]["dirs"] == []
    assert not paths.config_dir().exists()
    paths.config_dir().mkdir()
    user = paths.config_dir() / "config.toml"
    user.write_text('[tiers.light.codex]\nmodel = "custom-model"\n', encoding="utf-8")
    cfg = config.load()
    assert config.config_path() == user
    assert cfg["tiers"]["light"]["codex"] == {"model": "custom-model", "effort": "low"}
    assert cfg["providers"] == ["claude", "codex"]
    override = storage / "explicit.toml"
    override.write_text('\ufeff[chat]\nmin_tier = "heavy"\n', encoding="utf-8")
    monkeypatch.setenv("AI_ROUTER_CONFIG", str(override))
    assert config.config_path() == override
    assert config.load()["chat"]["min_tier"] == "heavy"
    assert config.load()["tiers"]["light"]["codex"]["model"] != "custom-model"
    monkeypatch.setenv("AI_ROUTER_CONFIG", str(storage / "missing.toml"))
    with pytest.raises(FileNotFoundError):
        config.load()


def test_invalid_user_config_fails_instead_of_using_defaults(storage):
    paths.config_dir().mkdir()
    user = paths.config_dir() / "config.toml"
    user.write_text("chat = false\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chat"):
        config.load()
    user.write_text("[chat\n", encoding="utf-8")
    with pytest.raises(ValueError):
        config.load()


def test_tui_watches_user_config_creation_and_updates(storage):
    before = tui._stamp()
    paths.config_dir().mkdir()
    user = paths.config_dir() / "config.toml"
    user.write_text('[chat]\nmin_tier = "heavy"\n', encoding="utf-8")
    assert tui._stamp() != before
    assert user in tui._watched() and config.PACKAGE_CONFIG in tui._watched()
    assert tui._broken() is None
    user.write_text("[chat\n", encoding="utf-8")
    assert tui._broken().startswith("config:")


@pytest.fixture
def legacy(storage):
    root = storage / "legacy"
    logs = root / "logs"
    logs.mkdir(parents=True)
    files = {"routes.jsonl": b"route\n", "labels.jsonl": b"label\n", "quota.json": b"{}",
             "ml_vectors.npz": b"vectors", "ml_settings.json": b"{}", "ml_threshold.json": b"{}",
             "ml_audit.jsonl": b"audit\n", "ml_history.jsonl": b"history\n",
             "style.md": b"short", "style_state.json": b"{}", "replies.jsonl": b"reply\n",
             "attachments/a.txt": b"attachment", "models/encoder/model.onnx": b"weights",
             "nested/run.log": b"log", "ml_teach.lock": b"1234", "models/encoder/model.part": b"partial"}
    for name, content in files.items():
        file = logs / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content)
    old_config = root / "airouter" / "config.toml"
    old_config.parent.mkdir()
    old_config.write_text('[chat]\nmin_tier = "heavy"\n', encoding="utf-8")
    return logs


def test_migration_preview_is_read_only_and_maps_every_category(legacy):
    entries = paths.migrate_data(legacy)
    targets = {e.source.relative_to(legacy).as_posix(): e.target for e in entries if e.source.is_relative_to(legacy)}
    assert targets["models/encoder/model.onnx"] == paths.model_dir() / "encoder" / "model.onnx"
    assert targets["attachments/a.txt"] == attach.folder() / "a.txt"
    assert targets["nested/run.log"] == paths.log_dir() / "nested" / "run.log"
    assert targets["ml_vectors.npz"] == learn.vectors_path()
    assert targets["quota.json"] == config.quota_path()
    assert len([e for e in entries if e.status == "skipped"]) == 2
    assert all(e.status in {"copy", "skipped"} for e in entries)
    assert not paths.data_dir().exists() and not paths.config_dir().exists()


def test_migration_copies_preserves_original_and_is_idempotent(legacy):
    entries = paths.migrate_data(legacy, apply=True)
    copied = [e for e in entries if e.status == "copied"]
    assert len(copied) == 15
    assert all(e.source.read_bytes() == e.target.read_bytes() for e in copied)
    assert all(e.target.stat().st_mtime_ns == e.source.stat().st_mtime_ns for e in copied)
    assert config.load()["chat"]["min_tier"] == "heavy"
    times = {e.target: e.target.stat().st_mtime_ns for e in copied}
    assert all(e.status in {"unchanged", "skipped"} for e in paths.migrate_data(legacy, apply=True))
    assert times == {p: p.stat().st_mtime_ns for p in times}
    assert not list(paths.data_dir().rglob("*.part"))


def test_conflict_prevents_all_copying_and_never_overwrites(legacy):
    target = paths.log_dir() / "labels.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text("existing", encoding="utf-8")
    entries = paths.migrate_data(legacy, apply=True)
    assert any(e.status == "conflict" and e.target == target for e in entries)
    assert target.read_text(encoding="utf-8") == "existing"
    assert not config.log_path().exists() and not paths.config_dir().exists()


def test_migration_honors_overrides_including_custom_filenames(legacy, storage, monkeypatch):
    monkeypatch.setenv("AI_ROUTER_LOG", str(storage / "route" / "custom.jsonl"))
    monkeypatch.setenv("AI_ROUTER_LABELS", str(storage / "learn" / "custom.jsonl"))
    monkeypatch.setenv("AI_ROUTER_ATTACH", str(storage / "images"))
    monkeypatch.setenv("AI_ROUTER_MODELS", str(storage / "weights"))
    entries = paths.migrate_data(legacy, apply=True)
    assert not any(e.status in {"error", "conflict"} for e in entries)
    assert config.log_path().read_bytes() == b"route\n"
    assert learn.labels_path().read_bytes() == b"label\n"
    assert learn.vectors_path().read_bytes() == b"vectors"
    assert (attach.folder() / "a.txt").read_bytes() == b"attachment"
    assert (paths.model_dir() / "encoder" / "model.onnx").read_bytes() == b"weights"


def test_migration_reports_permission_failure_without_changing_source(legacy, monkeypatch):
    original = paths._digest

    def deny(path):
        if path.name == "labels.jsonl":
            raise PermissionError("denied")
        return original(path)

    monkeypatch.setattr(paths, "_digest", deny)
    entries = paths.migrate_data(legacy, apply=True)
    assert any(e.status == "error" and "denied" in e.detail for e in entries)
    assert not paths.data_dir().exists()
    assert (legacy / "labels.jsonl").read_bytes() == b"label\n"


def test_migration_rejects_targets_inside_source_and_colliding_overrides(legacy, monkeypatch):
    monkeypatch.setenv("AI_ROUTER_DATA_DIR", str(legacy / "new"))
    assert any(e.status == "error" for e in paths.migrate_data(legacy, apply=True))
    assert not (legacy / "new").exists()
    monkeypatch.setenv("AI_ROUTER_LOG", str(legacy.parent / "same.jsonl"))
    monkeypatch.setenv("AI_ROUTER_LABELS", str(legacy.parent / "same.jsonl"))
    assert any(e.status == "conflict" for e in paths.migrate_data(legacy))


def test_migration_checks_racing_target_without_overwriting(legacy, monkeypatch):
    original = paths._publish
    raced = []

    def race(source, target):
        if not raced:
            raced.append(target)
            target.write_bytes(b"other writer")
        return original(source, target)

    monkeypatch.setattr(paths, "_publish", race)
    entries = paths.migrate_data(legacy, apply=True)
    assert any(e.status == "conflict" for e in entries)
    assert raced[0].read_bytes() == b"other writer"
    assert not list(paths.data_dir().rglob("*.part"))


def test_migration_reports_missing_explicit_source(storage):
    entries = paths.migrate_data(storage / "missing", apply=True)
    assert any(e.status == "error" and "tidak ada" in e.detail for e in entries)
    assert not paths.data_dir().exists()


def test_migration_rejects_file_directory_destination_collision(legacy, monkeypatch):
    monkeypatch.setenv("AI_ROUTER_MODELS", str(config.log_path()))
    entries = paths.migrate_data(legacy, apply=True)
    assert any(e.status == "conflict" and "direktori" in e.detail for e in entries)
    assert not paths.data_dir().exists()


def test_migration_skips_source_reparse_points(legacy, monkeypatch):
    original = paths._reparse
    excluded = legacy / "attachments"
    monkeypatch.setattr(paths, "_reparse", lambda p: p == excluded or original(p))
    entries = paths.migrate_data(legacy, apply=True)
    assert any(e.source == excluded and e.status == "skipped" for e in entries)
    assert not attach.folder().exists()
    assert config.log_path().read_bytes() == b"route\n"


def test_migration_refuses_target_reparse_points(legacy, monkeypatch):
    original = paths._reparse
    paths.data_dir().mkdir()
    monkeypatch.setattr(paths, "_reparse", lambda p: p == paths.data_dir() or original(p))
    entries = paths.migrate_data(legacy, apply=True)
    assert any(e.status == "error" and "reparse" in e.detail for e in entries)
    assert not config.log_path().exists()
    assert not paths.config_dir().exists()


def test_migration_does_not_replace_explicit_config(legacy, storage, monkeypatch):
    selected = storage / "explicit.toml"
    selected.write_text('[chat]\nmin_tier = "light"\n', encoding="utf-8")
    monkeypatch.setenv("AI_ROUTER_CONFIG", str(selected))
    entries = paths.migrate_data(legacy, apply=True)
    assert all(e.target != selected for e in entries)
    assert not paths.config_dir().exists()
    assert config.load()["chat"]["min_tier"] == "light"


def test_migration_cli_conflict_returns_failure(legacy, capsys):
    config.log_path().parent.mkdir(parents=True)
    config.log_path().write_bytes(b"existing")
    assert cli.main(["--migrate-data", "--migrate-from", str(legacy), "--apply"]) == 1
    assert "belum selesai" in capsys.readouterr().out
    assert config.log_path().read_bytes() == b"existing"


def test_migration_detects_source_changes_and_leaves_no_partial_target(legacy, monkeypatch):
    original = paths.migration_plan

    def change(source):
        plan = original(source)
        next(e.source for e in plan if e.status == "copy").write_bytes(b"changed")
        return plan

    monkeypatch.setattr(paths, "migration_plan", change)
    entries = paths.migrate_data(legacy, apply=True)
    assert any(e.status == "error" and "berubah" in e.detail for e in entries)
    assert not any(e.target.exists() for e in entries if e.status == "error")
    assert not list(paths.data_dir().rglob("*.part"))


def test_migration_cli_defaults_to_preview_and_does_not_load_config(legacy, monkeypatch, capsys):
    monkeypatch.setattr(config, "load", lambda: pytest.fail("migration must not load config"))
    assert cli.main(["--migrate-data", "--migrate-from", str(legacy)]) == 0
    assert "tanpa menulis" in capsys.readouterr().out
    assert not paths.data_dir().exists()
    assert cli.main(["--migrate-data", "--migrate-from", str(legacy), "--apply"]) == 0
    assert "Migrasi selesai" in capsys.readouterr().out


@pytest.mark.parametrize("args", [["--apply"], ["--migrate-from", "logs"],
                                 ["--migrate-data", "--apply", "-n"], ["--migrate-data", "hello"]])
def test_migration_cli_rejects_ambiguous_actions(args):
    with pytest.raises(SystemExit) as error:
        cli._parse(args)
    assert error.value.code == 2
