import json

from ..security import SecurityPolicy
from .common import node_script


def executable() -> list[str]:
    return node_script("codex", "@openai", "codex", "bin", "codex.js")


def build(model: str, effort: str, prompt: str, interactive: bool,
          policy: SecurityPolicy) -> tuple[list[str], str | None]:
    effort_config = ["-c", f"model_reasoning_effort={effort}"]
    roots = [arg for root in policy.extra_write_roots for arg in ("--add-dir", str(root))]
    if interactive:
        security = ["-s", policy.sandbox, "-a", policy.approval_policy,
                    "-C", str(policy.workspace)] + roots
        return executable() + ["-m", model] + effort_config + security + ([prompt] if prompt else []), None
    security = ["-s", policy.sandbox, "-C", str(policy.workspace),
                "-c", f'approval_policy="{policy.approval_policy}"'] + roots
    return executable() + ["exec", "-m", model] + effort_config + [
        "--skip-git-repo-check", "--ephemeral"] + security + ["-"], prompt


def server_command(policy: SecurityPolicy, roots: list[str]) -> list[str]:
    encoded = json.dumps(roots)
    return executable() + ["app-server", "--listen", "stdio://", "-c", f'sandbox_mode="{policy.sandbox}"',
                           "-c", f'approval_policy="{policy.approval_policy}"',
                           "-c", f"sandbox_workspace_write.writable_roots={encoded}",
                           "-c", f"sandbox_workspace_write.network_access={str(policy.network_allowed).lower()}"]


def chat_command(model: str, effort: str, session: str | None, images: list[str],
                 policy: SecurityPolicy, roots: list[str]) -> list[str]:
    common = ["-m", model, "-c", f"model_reasoning_effort={effort}", "--json", "--skip-git-repo-check"]
    common += [arg for image in images for arg in ("-i", image)]
    common += ["-c", f"sandbox_workspace_write.writable_roots={json.dumps(roots)}"]
    common += ["-c", f"sandbox_workspace_write.network_access={str(policy.network_allowed).lower()}"]
    common += ["-c", f'approval_policy="{policy.approval_policy}"']
    if session:
        return executable() + ["exec", "resume"] + common + [
            "-c", f'sandbox_mode="{policy.sandbox}"', session, "-"]
    return executable() + ["exec"] + common + ["-s", policy.sandbox, "-C", str(policy.workspace), "-"]

