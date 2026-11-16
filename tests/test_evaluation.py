import json

from airouter import cli, config, evaluation, learn


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _row(row_id, prompt, tier, verified=True):
    return {"id": row_id, "prompt": prompt, "gold_tier": tier, "task_group": "synthetic",
            "project_id": "holdout", "source": "test", "label_verified": verified,
            "created_at": "2026-01-01"}


def test_evaluation_reports_confusion_metrics_and_saves_artifact(tmp_path):
    dataset = tmp_path / "holdout.jsonl"
    _write(dataset, [
        _row("l1", "apa itu quantum banana?", "light"),
        _row("m1", "fix bug quantum banana function", "medium"),
        _row("h1", "rancang arsitektur quantum banana", "heavy"),
    ])
    report, saved = evaluation.run(dataset, config.load())
    assert report["valid_holdout"] and report["classification_accuracy"] == 1
    assert report["macro_f1"] == 1 and report["under_routing_rate"] == 0
    assert report["confusion"]["heavy"]["heavy"] == 1
    assert saved.exists() and json.loads(saved.read_text(encoding="utf-8"))["dataset_sha256"]


def test_evaluation_marks_training_leakage_invalid(tmp_path):
    trained = json.loads(learn.SEED.read_text(encoding="utf-8").splitlines()[0])
    dataset = tmp_path / "leaked.jsonl"
    _write(dataset, [_row("leaked", trained["text"], trained["tier"])])
    report = evaluation.evaluate(evaluation.load_dataset(dataset), config.load())
    assert not report["valid_holdout"] and report["leaked_ids"] == ["leaked"]


def test_unverified_labels_do_not_form_a_valid_holdout(tmp_path):
    dataset = tmp_path / "unverified.jsonl"
    _write(dataset, [_row("one", "apa itu unique holdout phrase", "light", verified=False)])
    report = evaluation.evaluate(evaluation.load_dataset(dataset), config.load())
    assert not report["valid_holdout"] and report["verified_labels"] == 0


def test_dataset_validation_rejects_duplicate_ids(tmp_path):
    dataset = tmp_path / "bad.jsonl"
    _write(dataset, [_row("same", "one", "light"), _row("same", "two", "medium")])
    try:
        evaluation.load_dataset(dataset)
    except ValueError as exc:
        assert "duplikat" in str(exc)
    else:
        raise AssertionError("duplicate dataset id accepted")


def test_eval_routing_cli_returns_nonzero_for_invalid_holdout(tmp_path, capsys):
    dataset = tmp_path / "unverified.jsonl"
    _write(dataset, [_row("one", "apa itu unique cli phrase", "light", verified=False)])
    assert cli.main(["--eval-routing", str(dataset)]) == 1
    shown = json.loads(capsys.readouterr().out)
    assert shown["valid_holdout"] is False


def _queue(prompts):
    path = learn.prompt_queue_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"prompt": p}) + "\n" for p in prompts), encoding="utf-8")


def test_build_holdout_freezes_model_labels_outside_training(tmp_path):
    trained = json.loads(learn.SEED.read_text(encoding="utf-8").splitlines()[0])["text"]
    _queue([trained, "rancang ulang zebra pipeline", "apa itu zebra", "/help", "apa itu zebra"])
    seen = []

    def labeler(prompts, model):
        seen.append((list(prompts), model))
        return ["heavy" if "rancang" in p else "light" for p in prompts]

    summary = evaluation.build_holdout(config.load(), 100, labeler_fn=labeler)
    assert seen == [(["rancang ulang zebra pipeline", "apa itu zebra"], "opus")]
    assert summary["samples"] == 2 and len(summary["spot_check_ids"]) == 2
    path = evaluation.holdout_path()
    assert path.is_relative_to(tmp_path / "data")
    rows = evaluation.load_dataset(path)
    assert all(row["label_model"] == "opus" and row["label_verified"] is False for row in rows)
    report = evaluation.evaluate(rows, config.load())
    assert report["model_labels"] == 2 and report["valid_holdout"]
    assert learn.untaught(learn.Learner.load()) == []
    try:
        evaluation.build_holdout(config.load(), 100, labeler_fn=labeler)
    except ValueError as exc:
        assert "dibekukan" in str(exc)
    else:
        raise AssertionError("frozen holdout overwritten")


def test_holdout_labeler_must_differ_from_teacher():
    cfg = config.load()
    cfg["classifier"]["holdout_labeler"] = cfg["classifier"]["teacher"]
    _queue(["apa itu zebra"])
    try:
        evaluation.build_holdout(cfg, 100, labeler_fn=lambda p, m: ["light"] * len(p))
    except ValueError as exc:
        assert "berbeda" in str(exc)
    else:
        raise AssertionError("teacher accepted as holdout labeler")


def test_teacher_labels_do_not_count_as_valid_holdout(tmp_path):
    dataset = tmp_path / "teacher.jsonl"
    row = _row("t", "apa itu unique teacher phrase", "light", verified=False)
    _write(dataset, [{**row, "label_source": "model", "label_model": "sonnet"}])
    report = evaluation.evaluate(evaluation.load_dataset(dataset), config.load())
    assert not report["valid_holdout"] and report["model_labels"] == 0


def test_spot_check_is_stable_and_detects_tampering(capsys):
    _queue([f"apa itu zebra nomor {i}" for i in range(30)])
    summary = evaluation.build_holdout(config.load(), 100, labeler_fn=lambda p, m: ["light"] * len(p))
    assert len(summary["spot_check_ids"]) == 20
    assert cli.main(["--spot-check"]) == 0
    assert len([line for line in capsys.readouterr().out.splitlines() if "[light]" in line]) == 20
    with evaluation.holdout_path().open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert cli.main(["--spot-check"]) == 2
