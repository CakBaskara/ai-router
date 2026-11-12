import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

from . import config, learn
from .learn import TIERS, classify


REQUIRED = {"id", "prompt", "gold_tier", "task_group", "source", "label_verified"}


def load_dataset(path: Path) -> list[dict]:
    rows = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"baris {number}: JSON tidak valid") from exc
        missing = REQUIRED - row.keys() if isinstance(row, dict) else REQUIRED
        if missing:
            raise ValueError(f"baris {number}: field wajib kurang: {', '.join(sorted(missing))}")
        if row["gold_tier"] not in TIERS:
            raise ValueError(f"baris {number}: gold_tier tidak valid")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in seen:
            raise ValueError(f"baris {number}: id kosong atau duplikat")
        if not isinstance(row["prompt"], str) or not row["prompt"].strip():
            raise ValueError(f"baris {number}: prompt kosong")
        seen.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError("dataset kosong")
    return rows


def _normalized(text: str) -> str:
    return " ".join(learn.features(text))


def leakage_ids(rows: list[dict], model: learn.Learner) -> list[str]:
    trained = {_normalized(row["text"]) for row in model.rows}
    return [row["id"] for row in rows if _normalized(row["prompt"]) in trained]


def _macro_f1(confusion: dict[str, Counter]) -> float:
    scores = []
    for tier in TIERS:
        true_positive = confusion[tier][tier]
        false_positive = sum(confusion[gold][tier] for gold in TIERS if gold != tier)
        false_negative = sum(confusion[tier][guess] for guess in TIERS if guess != tier)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0)
    return sum(scores) / len(scores)


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if not total:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [round(max(0, center - margin), 4), round(min(1, center + margin), 4)]


def _commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.decode("ascii", "ignore").strip() or None if result.returncode == 0 else None


def evaluate(rows: list[dict], cfg: dict) -> dict:
    model = learn.Learner.load(learn.get_encoder(wait=False) if cfg["classifier"].get("embed", True) else None)
    leaked = leakage_ids(rows, model)
    confusion = {tier: Counter() for tier in TIERS}
    ml_decisions = 0
    predicted = []
    floor = cfg.get("chat", {}).get("min_tier") or "light"
    for row in rows:
        verdict = classify(row["prompt"], cfg.get("rules", {}))
        tier = verdict.tier
        if verdict.ambiguous:
            guess, _, _ = learn.judge(row["prompt"], cfg, lambda _: None, use_llm=False, wait=False)
            if guess:
                tier = guess
                ml_decisions += 1
        effective = tier if TIERS.index(tier) >= TIERS.index(floor) else floor
        confusion[row["gold_tier"]][tier] += 1
        predicted.append((row, tier, effective))
    total = len(rows)
    correct = sum(confusion[tier][tier] for tier in TIERS)
    under = sum(TIERS.index(tier) < TIERS.index(row["gold_tier"]) for row, tier, _ in predicted)
    effective_under = sum(TIERS.index(tier) < TIERS.index(row["gold_tier"]) for row, _, tier in predicted)
    heavy = sum(row["gold_tier"] == "heavy" for row, _, _ in predicted)
    severe = sum(row["gold_tier"] == "heavy" and tier == "light" for row, tier, _ in predicted)
    verified = sum(row.get("label_verified") is True for row in rows)
    return {
        "samples": total,
        "verified_labels": verified,
        "valid_holdout": not leaked and verified == total,
        "leaked_ids": leaked,
        "classification_accuracy": round(correct / total, 4),
        "accuracy_95ci": _wilson(correct, total),
        "macro_f1": round(_macro_f1(confusion), 4),
        "under_routing_rate": round(under / total, 4),
        "effective_under_routing_rate": round(effective_under / total, 4),
        "severe_under_routing": round(severe / heavy, 4) if heavy else None,
        "ml_coverage": round(ml_decisions / total, 4),
        "confusion": {gold: {guess: confusion[gold][guess] for guess in TIERS} for gold in TIERS},
        "quality_success_rate": None,
        "quota_per_successful_task": None,
        "latency_p95": None,
    }


def run(path: Path, cfg: dict) -> tuple[dict, Path]:
    raw = path.read_bytes()
    rows = load_dataset(path)
    result = {
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": str(path.resolve()),
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "git_commit": _commit(),
        "classifier_version": learn.Learner.load().backend,
        **evaluate(rows, cfg),
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    target = config.log_path().with_name(f"routing_eval_{stamp}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, target
