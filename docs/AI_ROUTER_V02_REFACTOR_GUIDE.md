# AI Router v0.2 — Implementation Plan

Repository: https://github.com/CakBaskara/ai-router  
Basis review: branch `main`, inspected 2 November 2026.  
Status: **implementation instructions, not an applied patch**. Repository checkout and provider CLIs were not available in this execution environment; validate flags against installed CLI versions and run tests locally before merging.

## Contents

- [Progress](#progress)
- [Goals and architectural constraints](#goals-and-architectural-constraints)
- [PR-01 — Portable storage and modularization](#pr-01--portable-storage-and-modularization)
- [PR-02 — Security policies and approval behavior](#pr-02--security-policies-and-approval-behavior)
- [PR-03 — Safe failover state machine](#pr-03--safe-failover-state-machine)
- [PR-04 — Per-turn complexity and conversation continuity](#pr-04--per-turn-complexity-and-conversation-continuity)
- [PR-05 — Optimize routing for real opportunity cost](#pr-05--optimize-routing-for-real-opportunity-cost)
- [PR-06 — Leakage-safe independent evaluation](#pr-06--leakage-safe-independent-evaluation)
- [Regression / acceptance checklist](#regression--acceptance-checklist)
- [Reference links](#reference-links)

## Progress

Audited against the working tree on 6 November 2026 (HEAD `20a4861`).

| PR | Status | State of the code |
|---|---|---|
| PR-01 | Done (`21971c0`) | `paths.py` with `platformdirs`; config order `AI_ROUTER_CONFIG`, user file, bundled defaults with merge; `ai --migrate-data` preview and `--apply`, never overwrites, idempotent, skips reparse points; 21 tests in `tests/test_paths.py`. `chat_roots()` no longer adds the repository root. Deliberate differences: models live under the data directory rather than a cache directory, and the other modules in the layout below wait for the PR that needs them. |
| PR-02 | Implemented, provider validation remains | `security.py` defines four modes and explicit roots. The default is `full-access` (owner decision, 11 November 2026): Claude `bypassPermissions`, Codex `danger-full-access` + approval `never`, no prompts. In the stricter modes the current workspace is the primary sandbox; attachments, `chat.dirs`, and the package are not implicit writable roots. Claude and all three Codex approval request types reach the same default-deny UI. One-shot approval mode becomes read-only because it has no approval UI. Unit coverage verifies command mappings and request handling; destructive live-provider probes have not been run. |
| PR-03 | Implemented | `failover.py` records dispatch phase and before/after workspace snapshots. Only a pre-dispatch spawn failure can fail over automatically; cancellation, unknown execution, and observed file changes stop. One-shot mode follows the same dispatch rule. |
| PR-04 | Implemented | `routing_policy.py` separates turn, context, and effective tiers. Explicit new topics can downgrade; references, short ambiguous follow-ups, and lexical continuity retain context. All fields and the reason are logged. |
| PR-05 | Initial production policy | `optimizer.py` protects a fresh 5-hour quota reserve, deprioritizes depleted providers, and treats missing/stale telemetry as unverified before falling back to success counts. Numeric turn usage is logged. Quality and latency ranking remain future work rather than fabricated inputs. |
| PR-06 | Evaluation path implemented | `ai --eval-routing DATASET` validates a frozen JSONL holdout, detects exact normalized training leakage, reports confusion/accuracy/macro-F1/under-routing/coverage with an accuracy interval, and stores a dataset-hashed JSON artifact. No repository holdout is claimed as human-verified; release owners must supply and freeze one. |

Next release task: run controlled live-provider security probes and curate a sufficiently large, human-verified holdout.

## Goals and architectural constraints

Preserve existing `/model`, pinned provider/model, CLI one-shot, Textual TUI, ongoing session/carryover, model review and learning flows. Do not let model-generated text override sandbox or approval policy. Treat `approval-required` as a higher-level policy layered on top of the actual sandbox, not a native third sandbox mode.

**Default security mode is fixed: `full-access`.** The owner wants AI Router to behave like the Claude Code and Codex IDE extensions in full-access mode: no permission prompts and no folder limits. The previous default `approval-required` made Claude ask (or silently fail) for every shell command and every read outside the current project, which the owner rejected (11 November 2026). Do not change the default back to `approval-required`, `workspace-write` or `read-only`, do not add prompts or folder limits to `full-access`, and do not restore the default-deny wording below as a global rule. The stricter modes stay available as opt-in. `tests/test_security.py::test_full_access_is_the_default_mode` guards this; changing that test needs the owner's explicit approval.

Suggested PR order:

1. PR-01 runtime path migration, config defaults and state safety.
2. PR-02 security modes, provider adapters and approval UI.
3. PR-03 safe failover state machine and change detection.
4. PR-04 per-turn vs conversation tier routing.
5. PR-05 utility-based provider choice with real instrumentation.
6. PR-06 holdout evaluation and CI regression gates.

Running all PRs simultaneously is discouraged because `chat.py` and `dispatch.py` are touched by most changes.

## PR-01 — Portable storage and modularization

### Current implementation

- `airouter/config.py`: `REPO_ROOT`, `PACKAGE_CONFIG`, `load()`, `log_path()`, `quota_path()`.
- `airouter/learn.py`: `model_folder()`, `labels_path()`, `style_path()`, `replies_path()` and derived paths.
- `airouter/attach.py`: `folder()`.
- `airouter/dispatch.py`: `chat_roots()` currently includes `config.REPO_ROOT` automatically. **Remove from writable roots**; this is separate from moving runtime data.

### Recommended target structure on Windows

```
%APPDATA%/AI-Router/config.toml
%LOCALAPPDATA%/AI-Router/logs/routes.jsonl
%LOCALAPPDATA%/AI-Router/logs/labels.jsonl
%LOCALAPPDATA%/AI-Router/logs/quota.json
%LOCALAPPDATA%/AI-Router/attachments/
%LOCALAPPDATA%/AI-Router/models/
```

Keep package assets such as `seed_labels.jsonl` inside the installed package; never write there. Linux/macOS should use platform-appropriate locations. Add `platformdirs>=4` to `pyproject.toml`.

Create `airouter/paths.py`:

```python
import os
from pathlib import Path
from platformdirs import user_config_dir, user_data_dir, user_cache_dir

APP_NAME = "AI-Router"

def config_dir() -> Path:
    return Path(user_config_dir(APP_NAME, appauthor=False))

def data_dir() -> Path:
    return Path(os.environ.get("AI_ROUTER_DATA_DIR", user_data_dir(APP_NAME, appauthor=False)))

def cache_dir() -> Path:
    return Path(user_cache_dir(APP_NAME, appauthor=False))

def log_dir() -> Path:
    return data_dir() / "logs"

def attachment_dir() -> Path:
    return data_dir() / "attachments"

def model_dir() -> Path:
    return Path(os.environ.get("AI_ROUTER_MODELS", cache_dir() / "models"))
```

Integrate `config.log_path()`, `learn.labels_path()`, `learn.model_folder()`, and `attach.folder()` using these functions, preserving their existing individual environment-variable overrides. `config.load()` selection order: `AI_ROUTER_CONFIG` if specified; then user configuration if it exists; finally bundled default. Do not `mkdir` until writing data.

Migration: add explicit `ai --migrate-data` (dry-run first). Copy `logs/` recursively to target and report conflicts; never overwrite existing target or delete original automatically. Back up data; skip `*.lock` and incomplete `*.part` as appropriate. Handle permissions and report conflicts. Preserve labels, vector cache, thresholds, settings, style and quota data. An idempotence test must pass.

### Modularize progressively

```
airouter/
  paths.py               # directories, state migration
  security.py            # SecurityPolicy, root checks
  failover.py            # ExecutionPhase, RetryDecision
  routing_policy.py      # per-turn and conversation decisions
  optimizer.py           # model/provider utility & telemetry
  evaluation.py          # holdout metrics and artifact format
  providers/
    claude.py            # Claude command/stream adapter
    codex.py             # Codex app-server adapter
```

Do not move all 900+ lines at once: introduce modules with narrow interfaces; keep backward-compatible functions in `chat.py` / `dispatch.py` initially.

## PR-02 — Security policies and approval behavior

### Definitions

- `full-access` (default): no sandbox and no approval prompts. Claude runs with `--permission-mode bypassPermissions`; Codex runs with `danger-full-access` and approval policy `never`. Every folder and the network are reachable; `allow_network`, `extra_write_roots` and `chat.dirs` do not limit it.
- `read-only`: no writing to active project; provider allowed only read tools/operations.
- `workspace-write`: non-destructive changes in explicitly selected project workspace permitted without additional prompt; any external/path escape remains blocked or escalated.
- `approval-required`: **request review before change**. Sandbox remains workspace-restricted; commands, file changes, and external side effects must be subject to explicit approval. A read-only *planning* stage followed by an isolated change preview and patch acceptance is the strongest cross-provider design.

Create `airouter/security.py`:

```python
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

class Mode(str, Enum):
    FULL_ACCESS = "full-access"
    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    APPROVAL_REQUIRED = "approval-required"

@dataclass(frozen=True)
class SecurityPolicy:
    mode: Mode
    workspace: Path
    network_allowed: bool = False

    def __post_init__(self):
        if not self.workspace.is_dir():
            raise ValueError("workspace must exist")

    def inside_workspace(self, path: Path) -> bool:
        return path.resolve().is_relative_to(self.workspace.resolve())
```

**Important:** `inside_workspace()` is validation for paths under your code's control, not a security boundary against a subprocess. Sandboxing must come from the provider/OS. Symlink and reparse-point escapes, shell commands, external tools, and network effects require enforcement beyond a Python path check.

In `config.toml`, introduce:

```toml
[security]
mode = "full-access"  # default; stricter: workspace-write, approval-required, read-only
allow_network = false
confirm_destructive = true
extra_write_roots = []

[chat]
min_tier = "light"
dirs = []  # optional read-only directories (Claude Read allow rules in the stricter modes), NOT writable roots
```

In `dispatch.py`, change `chat_roots()` to obtain **explicit writable roots from the security policy** and use `Path.cwd().resolve()` as the primary workspace. Never add `config.REPO_ROOT` or attachment storage automatically as writable roots. Attachment files should be read-only or passed as input data. Change function signatures `chat_cmd(..., policy)`, `codex_server_cmd(policy)`, and `CodexProc(..., policy)`; pass through `Chat._run` and `ClaudeProc`.

Provider mapping (verify against each installed version):

| AI Router policy | Claude Code | Codex CLI / App-server |
|---|---|---|
| full-access (default) | `--permission-mode bypassPermissions`; no `--disallowedTools`, no `--restricted` | `-s danger-full-access`, `approval_policy="never"`; app-server thread `sandbox: "danger-full-access"`, `approvalPolicy: "never"` |
| read-only | `--permission-mode plan` and a deliberately small available-tools set (`--tools`); no shell/write tools | `sandbox=readOnly` / `sandboxPolicy.type=readOnly` |
| workspace-write | `acceptEdits` only with independently enforced workspace boundary | `workspaceWrite` with one writable project root, network disabled |
| approval-required | `default`/manual plus a real `PermissionRequest` hook/UI; or safer two-phase isolated patch review | `workspaceWrite` + `approvalPolicy` and client-side approval requests; for **every edit** use a separate working copy + diff approval, since normal sandboxed edits need not trigger approval |

See official Claude permissions https://code.claude.com/docs/en/permissions and Codex app-server docs https://learn.chatgpt.com/docs/app-server .

`CodexProc._read()` currently rejects unknown JSON-RPC client requests. Add handling for:

- `item/commandExecution/requestApproval`
- `item/fileChange/requestApproval`
- `item/permissions/requestApproval`

In `approval-required` mode, do not auto-approve. The UI must show exact command/target/reason where supplied; scope approvals to correct thread+turn IDs; decline on timeout, cancellation, missing UI, or mismatch. For richer file-approval messages, look up the matching `fileChange` item/diff. Avoid `acceptForSession` by default.

**Strict approval design:** read-only planning -> create isolated ephemeral Git worktree/copy with sandboxed write permissions and no network -> produce patch and untracked files manifest -> show diff to user -> apply to target workspace only after approval -> run tests -> report. A Git worktree is not by itself a filesystem sandbox. Use provider/OS containment. For non-Git folders, use a copy; do not silently require a Git repository.

Tests:

0. `full-access` is the bundled default and maps to `bypassPermissions` / `danger-full-access` + `never` for chat, one-shot and app-server commands.
1. `read-only` prevents file edits from both Claude and Codex.
2. Workspace root does not include AI Router package directory or arbitrary `chat.dirs`.
3. Symlink/reparse point traversal cannot escape the enforced root.
4. Missing/declined approval does not modify active workspace.
5. TUI and `--plain` both ask or default-deny safely.
6. Changing mode recreates an incompatible provider session/process; no stale permission carries over.

## PR-03 — Safe failover state machine

### Present code

`airouter/chat.py:Chat.send()` loops through providers after failure. `airouter/cli.py:main()` has a separate one-shot failover loop. Preserve that behavior only when safe.

Create `airouter/failover.py`:

```python
from dataclasses import dataclass
from enum import Enum

class Phase(str, Enum):
    NOT_DISPATCHED = "not_dispatched"
    DISPATCHED = "dispatched"
    EFFECTS_OBSERVED = "effects_observed"
    FINISHED = "finished"
    UNKNOWN = "unknown"

@dataclass(frozen=True)
class AttemptResult:
    exit_code: int
    phase: Phase
    cancelled: bool = False
    file_changes_detected: bool = False
    external_effects_possible: bool = True


def auto_retry_allowed(result: AttemptResult) -> bool:
    return (
        not result.cancelled
        and result.phase is Phase.NOT_DISPATCHED
        and not result.file_changes_detected
        and not result.external_effects_possible
    )
```

`NOT_DISPATCHED` means there is **strong evidence** process/turn dispatch never happened (such as executable not found before spawn). An RPC timeout after a request was sent is `UNKNOWN`, not `NOT_DISPATCHED`. A stream disconnect is at least `DISPATCHED`; events indicating tool/file operations imply `EFFECTS_OBSERVED`. A successful LLM response can still have effects. A cancelled turn must never automatically fail over.

Before a write-capable turn, capture a workspace **baseline**; after failure capture a second one:

- Git status and both staged/unstaged diffs.
- Hash of currently tracked modified files and detected untracked files, compared against the baseline.
- For stronger coverage, snapshot all scoped files, including ignored files if safe to do so, with size/volume caps.
- Limit monitoring to the workspace; explain that this cannot detect network operations or changes outside the workspace.
- Do not assume a repository is clean initially or that identical `git status` means no modifications.

Decision table:

| State | Action |
|---|---|
| Failed to spawn, request never dispatched, no effects possible | Retry next provider automatically |
| Turn dispatched, no proof of clean non-execution | Stop; ask user |
| Local edits detected | Show diff; require explicit user decision before retry |
| External effects possible (git push, deployment, sending data, writing to API) | Never auto-retry; verify external state manually |
| User cancelled | Stop; no retry |

UI should display provider, error, phase, changed files, links/paths to diff; options: keep current changes, inspect diff, retry with explicit approval, abort. **Do not silently revert** preexisting user changes or the model's work.

Pseudocode integration in `Chat.send()`:

```python
baseline = workspace_snapshot(policy.workspace)
attempt = self.run_attempt(...)
if attempt.ok:
    return 0
new_state = workspace_snapshot(policy.workspace)
result = classify_attempt(attempt, baseline, new_state)
if auto_retry_allowed(result):
    continue
self.ui.report_unsafe_failover(result, diff=compare(baseline, new_state))
return attempt.exit_code or 1
```

`workspace_snapshot`, `classify_attempt`, and `report_unsafe_failover` are **new interfaces to implement**, not existing functions. If you cannot safely detect a phase, mark it `UNKNOWN` and stop. Apply the same safety policy to `cli.main()` one-shot.

Critical test cases: spawn failure auto-retries; RPC disconnect does not; edit then failure does not; cancellation does not; preexisting dirty files remain unchanged; identical status but changed content is detected; untracked files detected; external writes cannot be declared safe by Git alone.

## PR-04 — Per-turn complexity and conversation continuity

### Present code

`Chat.route()` uses `higher(self.tier, tier)` to retain the highest-ever tier, so it cannot auto-downgrade after a topic switch. `self.tier` also means the most recent selected tier in status and model selection.

Introduce `RoutingDecision` in `airouter/routing_policy.py`:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class RoutingDecision:
    turn_tier: str
    context_tier: str
    effective_tier: str
    related_to_previous: bool
    reason: str
```

Rules:

1. `turn_tier` is classified independently of prior tier.
2. `context_tier` is the minimum capability needed to continue active work; it should NOT be the maximum tier ever seen.
3. A new, self-contained question uses `turn_tier`, subject to `chat.min_tier` and user pin.
4. A dependent follow-up (`lanjutkan`, `perbaiki error tadi`, references to previous generated files) may inherit previous task tier.
5. A short ambiguous follow-up needs continuity; don't downgrade solely because it has three words.
6. An explicit pin is always honored; user-model selection is not overridden by automatic topic detection.
7. Record both tiers and why the context was inherited.

Initial implementation can use deterministic rules for explicit references and light lexical similarity; upgrade later to semantic similarity using the existing embedding encoder. Do not split a task based solely on a fixed time delay or one low cosine score. Keep session memory independent of tier: `transcript`, `sessions`, and `synced` already exist. Cross-provider context recap is limited to recent entries; audit summarization before increasing its size.

Example tests:

| Previous turn | New turn | Expected |
|---|---|---|
| Heavy: refactor parser | `lanjutkan dan perbaiki test tadi` | Heavy/related |
| Heavy: refactor parser | `apa itu git rebase?` | Light/new task |
| Medium: fix small function | `error-nya masih sama` | Medium/related |
| Heavy: refactor parser | `/model light` pinned | Light (user pin) |

Do not replace `self.tier` with `turn_tier` in every location blindly: inspect `Chat.prewarm()`, `Chat.correct()`, status, snapshot/restore and provider session reuse.

## PR-05 — Optimize routing for real opportunity cost

### Subscription warning

AI Router uses Claude Code and Codex subscriptions; the main constrained resource is **subscription quota**, not necessarily per-token API billing. A plan cannot assume a public API price maps to subscription depletion. Model cost should be interpreted as *opportunity cost*, measured from observed usage/quota trends where possible. `lineup()` currently uses success-count ordering via `usage_today()`. Distinguish **load balancing** from **quality-constrained quota optimization**.

Per candidate provider/model, collect:

- `eligible`: candidate permitted for `turn_tier` and security mode; CLI installed, configured, supported and not out-of-quota.
- `success_rate`: task-based success metric with uncertainty, ideally per task category.
- `quality_score`: held-out benchmark success and user correction rate; never treat teacher tier agreement as quality.
- `latency_p50/p95` and exponential moving average.
- `tokens_input`, `tokens_output`, cached tokens (raw numbers, not only formatted display text).
- `quota_5h_used`, `quota_week_used`, timestamp and freshness; mark stale/missing.
- `effective_quota_cost`: observed quota change per comparable task or per 1k tokens when reliably inferable.

Quality should be a hard threshold before utility comparison. A candidate that repeatedly fails hard tasks must not win merely because it is cheap. If telemetry is unavailable, fall back to current rule while explicitly reporting `cost_unverified`; do not fabricate a cost estimate.

Illustrative decision rule:

```
eligible = [m for m in models if quality_lower_bound(m, task) >= required_quality]
if not eligible: choose strongest reliable model or ask user
for m in eligible:
    utility[m] = (
        + 0.50 * normalized_quality[m]
        + 0.25 * quota_headroom[m]
        - 0.15 * normalized_latency[m]
        - 0.10 * opportunity_cost[m]
    )
choose max(utility)  # example weights only; tune against held-out data
```

**Prefer quota constraints over a raw weighted score:** avoid 5-hour quota below user-set reserve (e.g. 15%) unless there is no viable model, and use weekly budget as a separate constraint. Normalization, freshness penalties, task grouping, and sample-size confidence are required before trusting these weights.

Suggested changes:

- `airouter/optimizer.py`: `Telemetry`, `Candidate`, `rank_candidates()`.
- `chat.py`: `order(tier)` takes current task and calls `rank_candidates`, while pinned provider retains priority.
- `cli.py`: one-shot mode calls the same policy; do not maintain a separate optimizer.
- `config.py`: preserve `quota_left`; add timestamp freshness and a counter for failed attempts.
- `ClaudeTurn.feed()` and `CodexTurn.feed()`: retain numeric usage in addition to text display.
- Every route log includes `tier_predicted`, `tier_effective`, `route_reason`, `candidate_scores`, `estimated_quota_cost`, `actual_usage`, and `outcome` (without storing raw prompt by default).

A/B evaluation: compare count-based baseline against optimizer for a fixed suite with quality and latency checks; report quota deltas only when both sources have trustworthy usage readings.

## PR-06 — Leakage-safe independent evaluation

### Why current cross-validation is not enough

In `learn.py`, `Learner.cross_val()` produces CV predictions; `Learner.gate()`, `score()`, `tune()`, and `report()` reuse those results to choose thresholds/hyperparameters and report quality. This makes the reported scores vulnerable to optimistic selection bias. The runtime gate's learned thresholds are not equivalent to a frozen held-out evaluation.

Make `data/eval_holdout.jsonl` a distinct, **frozen** sample of 200–500 human-reviewed prompts across light/medium/heavy and several code/non-code topics. Avoid publishing private prompts; redaction or synthetic examples can be used for an open-source fixture. Each row should have stable `id`, `prompt`, `gold_tier`, `task_group`, `project_id`, `source`, `label_verified`, and `created_at`. Exclude these IDs and near-duplicates from `Learner.load()` training data.

Split by project/task group, not by individual prompt, so rephrasings of the same example do not leak across train and test. For continual learning, also evaluate an out-of-time slice from newly collected sessions that has not been used for tuning. Tune only on train/validation; run holdout once for release gate, not iteratively to guide hyperparameter changes.

Define metrics:

```
classification_accuracy = correct_tier / total
under_routing_rate = count(predicted_tier < gold_tier) / total
severe_under_routing = count(heavy -> light) / total_heavy
macro_f1 = average(F1_light, F1_medium, F1_heavy)
ml_coverage = ML_final_decisions / all_attempts
quality_success_rate = tasks_passed / tasks_executed
quota_per_successful_task = sum(measured_quota_consumption) / tasks_passed
latency_p95 = p95(end_to_end_seconds)
```

Separate model-predicted tier from final effective tier (which may have floors or pins). Report both. If only a subset is routed via ML, calculate conditional under-routing on those decisions too. Apply confidence intervals when n is small.

`quality_success_rate` needs real task outcomes (tests passing, patch accepted, or human-reviewed rubric) and must not be inferred solely from classifier gold labels. Benchmark tasks that mutate code must run inside disposable isolated projects; do not evaluate against a user's active workspace.

CLI addition: `ai --eval-routing --dataset tests/fixtures/routing_holdout.jsonl` prints confusion matrix, under-routing, severe misses, coverage, macro-F1, quality/latency/quota fields if available. Save JSON report under state directory with dataset hash, git commit hash, classifier version and run timestamp.

Do not set a 5% under-routing acceptance threshold until enough examples exist for statistically meaningful confidence; a sample of 20 hard tasks does not prove a <5% miss rate.

## Regression / acceptance checklist

- [x] All existing `tests/test_chat.py` and `tests/test_learn.py` cases adjusted and passing.
- [x] Tests with a fake runner and explicit safety policy, including cancel/inject/recap/restore.
- [ ] Tests on Windows 11 with current `claude --help`, `codex --help` and installed app-server protocol.
- [x] `/model` pin, model switch, `ai -n`, one-shot and `--plain` preserve behavior.
- [x] Default mode is `full-access` with no permission prompts for Claude or Codex.
- [x] Sandboxed modes give no writable access to the AI Router installation directory; `full-access` has no filesystem boundary by design.
- [x] No automatic failover after dispatched/unknown side effects.
- [x] Read-only mode maps both providers to non-writing policies; a controlled live-provider probe remains pending.
- [x] In `approval-required`, missing, mismatched, cancelled, or refused approval defaults to deny before the requested action.
- [x] Path migration idempotent; no user data deleted.
- [x] Holdout evaluation detects training leakage and rejects unverified labels; a release dataset is not bundled.
- [x] Router can downgrade after topic change without losing dependent follow-up context.
- [x] Missing/stale quota data causes safe fallback rather than fabricated 'cheapest' claims.

Recommended Windows commands:

```powershell
git switch -c refactor/ai-router-v02
py -m pip install -e . pytest platformdirs
py -m pytest -q
py -m compileall -q airouter
claude --help
codex --help
```

**Suggested implementation sequence:** Implement PR-01 paths first; then PR-02 security policy (default `full-access`, stricter modes opt-in); PR-03 safe failover; PR-04 tier reset; PR-05 telemetry/optimizer; and PR-06 holdout evaluation. Keep feature flags to turn off each new behavior independently while testing.

## Reference links

- `chat.py`: https://github.com/CakBaskara/ai-router/blob/main/airouter/chat.py
- `dispatch.py`: https://github.com/CakBaskara/ai-router/blob/main/airouter/dispatch.py
- `learn.py`: https://github.com/CakBaskara/ai-router/blob/main/airouter/learn.py
- `config.py`: https://github.com/CakBaskara/ai-router/blob/main/airouter/config.py
- `config.toml`: https://github.com/CakBaskara/ai-router/blob/main/airouter/config.toml
- Claude permissions: https://code.claude.com/docs/en/permissions
- Codex app-server: https://learn.chatgpt.com/docs/app-server
