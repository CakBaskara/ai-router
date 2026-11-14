import subprocess

from airouter import handoff


def test_handoff_refreshes_git_metadata(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q", "-b", "baseline")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com",
        "commit", "-q", "--allow-empty", "-m", "Baseline")
    checkpoint = handoff.record({}, tmp_path, "claude", "Review baseline", "Done", "completed")
    assert checkpoint["git"]["branch"] == "baseline"
    assert checkpoint["git"]["commit"] == git("rev-parse", "HEAD")
    assert checkpoint["git"]["status"] == ""

    git("switch", "-qc", "follow-up")
    (tmp_path / "evidence.txt").write_text("test build log", encoding="utf-8")
    preview = handoff.render(checkpoint, tmp_path, refresh=True)
    assert "Git branch: follow-up" in preview
    assert "?? evidence.txt" in preview
    assert "differs from the saved checkpoint" in preview
    assert "test build log" not in preview
