"""Checks for the GitHub Actions workflow. No secrets and no deployment."""

from __future__ import annotations

from src.data_loader import PROJECT_ROOT


def test_ci_workflow_runs_pytest_on_push_and_pull_request():
    text = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "on:" in text
    assert "push:" in text
    assert "pull_request:" in text
    assert "pip install -r requirements.txt" in text
    assert "python -m pytest" in text
    assert "secrets." not in text
    assert "deploy" not in text.lower()


def test_official_files_present_is_false_for_an_empty_directory(tmp_path, monkeypatch):
    from src import data_loader as dl

    monkeypatch.setattr(dl, "RAW_DIR_CANDIDATES", (tmp_path,))
    assert dl.official_files_present() is False
