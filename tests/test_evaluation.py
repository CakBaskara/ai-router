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
