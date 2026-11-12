import json
import tomllib

import pytest

from airouter import config, dispatch
from airouter.chat import ClaudeProc
from airouter.security import Mode, SecurityPolicy


def policy(tmp_path, mode=Mode.APPROVAL_REQUIRED, **kwargs):
    return SecurityPolicy(mode, tmp_path, **kwargs)


def test_security_policy_validates_mode_and_roots(tmp_path):
    cfg = {"security": {"mode": "invalid"}, "chat": {}}
    with pytest.raises(ValueError, match="security.mode"):
        SecurityPolicy.from_config(cfg, tmp_path)
    with pytest.raises(ValueError, match="write root"):
        SecurityPolicy(Mode.WORKSPACE_WRITE, tmp_path, extra_write_roots=(tmp_path / "missing",))


def test_inside_workspace_resolves_parent_traversal(tmp_path):
    workspace = tmp_path / "project"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    selected = policy(workspace)
    assert selected.inside_workspace(workspace / "src" / "file.py")
    assert not selected.inside_workspace(workspace / ".." / "outside" / "secret.txt")


def test_read_only_commands_disable_write_and_network_tools(tmp_path):
    selected = policy(tmp_path, Mode.READ_ONLY)
    claude = dispatch.chat_cmd("claude", "sonnet", "low", None, policy=selected)
    assert claude[claude.index("--permission-mode") + 1] == "plan"
    assert "--restricted" in claude and claude[claude.index("--tools") + 1] == "Read,Grep,Glob"
    codex = dispatch.chat_cmd("codex", "model", "low", None, policy=selected)
    assert codex[codex.index("-s") + 1] == "read-only"
    assert "sandbox_workspace_write.network_access=false" in codex


def test_approval_mode_routes_prompts_to_host_and_blocks_web_tools(tmp_path):
    selected = policy(tmp_path)
    claude = dispatch.chat_cmd("claude", "sonnet", "low", None, policy=selected)
    assert claude[claude.index("--permission-mode") + 1] == "manual"
    assert claude[claude.index("--permission-prompts") + 1] == "host"
    assert claude[claude.index("--disallowedTools") + 1] == "WebFetch,WebSearch"
    codex = dispatch.chat_cmd("codex", "model", "low", None, policy=selected)
    assert 'approval_policy="on-request"' in codex


def test_one_shot_without_an_approval_ui_falls_back_to_read_only(tmp_path):
    selected = policy(tmp_path)
    claude, _ = dispatch.build("claude", "sonnet", "low", "inspect", False, selected)
    codex, _ = dispatch.build("codex", "model", "low", "inspect", False, selected)
    assert claude[claude.index("--permission-mode") + 1] == "plan"
    assert codex[codex.index("-s") + 1] == "read-only"


def test_only_explicit_security_roots_are_writable(tmp_path):
    workspace = tmp_path / "project"
    extra = tmp_path / "shared"
    reads = tmp_path / "reference"
    attachments = tmp_path / "attachments"
    for item in (workspace, extra, reads, attachments):
        item.mkdir()
    selected = SecurityPolicy(Mode.WORKSPACE_WRITE, workspace, extra_write_roots=(extra,), read_roots=(reads,))
    roots = dispatch.chat_roots(selected)
    assert roots == [str(extra.resolve())]
    assert str(config.REPO_ROOT.resolve()) not in roots
    assert str(reads.resolve()) not in roots and str(attachments.resolve()) not in roots
    command = dispatch.codex_server_cmd(selected)
    setting = next(arg for arg in command if arg.startswith("sandbox_workspace_write.writable_roots="))
    assert json.loads(setting.split("=", 1)[1]) == roots


class _Input:
    def __init__(self):
        self.written = []

    def write(self, value):
        self.written.append(json.loads(value))

    def flush(self):
        pass


class _Proc:
    def __init__(self):
        self.stdin = _Input()

    def poll(self):
        return None


@pytest.mark.parametrize("decision", [True, False])
def test_claude_permission_request_is_returned_to_the_cli(decision):
    proc = ClaudeProc.__new__(ClaudeProc)
    proc.proc = _Proc()
    proc.approve = lambda shown, cancelled: decision
    proc._permission({
        "type": "control_request",
        "request_id": "request-1",
        "request": {"subtype": "can_use_tool", "tool_name": "Write",
                    "input": {"file_path": "x.txt"}},
    })
    response = proc.proc.stdin.written[0]
    answer = response["response"]["response"]
    assert answer["behavior"] == ("allow" if decision else "deny")


def test_policy_key_changes_when_mode_changes(tmp_path):
    assert policy(tmp_path, Mode.READ_ONLY).key != policy(tmp_path, Mode.WORKSPACE_WRITE).key


def test_workspace_write_keeps_destructive_approval_unless_disabled(tmp_path):
    assert policy(tmp_path, Mode.WORKSPACE_WRITE).approval_policy == "on-request"
    assert policy(tmp_path, Mode.WORKSPACE_WRITE, confirm_destructive=False).approval_policy == "never"


def test_full_access_is_the_default_mode(tmp_path):
    bundled = tomllib.loads(config.PACKAGE_CONFIG.read_text(encoding="utf-8-sig"))
    assert SecurityPolicy.from_config(bundled, tmp_path).mode is Mode.FULL_ACCESS
    assert SecurityPolicy.from_config({"chat": {}}, tmp_path).mode is Mode.FULL_ACCESS


def test_full_access_runs_both_providers_without_prompts_or_sandbox(tmp_path):
    selected = policy(tmp_path, Mode.FULL_ACCESS)
    assert selected.network_allowed and selected.approval_policy == "never"
    claude = dispatch.chat_cmd("claude", "sonnet", "low", None, policy=selected)
    assert claude[claude.index("--permission-mode") + 1] == "bypassPermissions"
    assert "--disallowedTools" not in claude and "--restricted" not in claude
    codex = dispatch.chat_cmd("codex", "model", "low", None, policy=selected)
    assert codex[codex.index("-s") + 1] == "danger-full-access" and 'approval_policy="never"' in codex
    server = dispatch.codex_server_cmd(selected)
    assert 'sandbox_mode="danger-full-access"' in server and 'approval_policy="never"' in server
    claude_once, _ = dispatch.build("claude", "sonnet", "low", "inspect", False, selected)
    codex_once, _ = dispatch.build("codex", "model", "low", "inspect", False, selected)
    assert claude_once[claude_once.index("--permission-mode") + 1] == "bypassPermissions"
    assert codex_once[codex_once.index("-s") + 1] == "danger-full-access"
    codex_live, _ = dispatch.build("codex", "model", "low", "", True, selected)
    assert codex_live[codex_live.index("-a") + 1] == "never"


def test_chat_dirs_reach_claude_as_read_access_in_every_mode(tmp_path):
    reads = tmp_path / "reference"
    reads.mkdir()
    full = SecurityPolicy(Mode.FULL_ACCESS, tmp_path, read_roots=(reads,))
    claude = dispatch.chat_cmd("claude", "m", "low", None, policy=full)
    assert claude[claude.index("--add-dir") + 1] == str(reads.resolve())
    strict = SecurityPolicy(Mode.APPROVAL_REQUIRED, tmp_path, read_roots=(reads,))
    claude = dispatch.chat_cmd("claude", "m", "low", None, policy=strict)
    assert "--add-dir" not in claude
    rule = claude[claude.index("--allowedTools") + 1]
    assert rule.startswith("Read(//") and rule.endswith(reads.resolve().as_posix().split(":", 1)[-1] + "/**)")
