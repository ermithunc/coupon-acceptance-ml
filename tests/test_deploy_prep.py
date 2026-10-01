"""Phase 20 checks. They do not call AWS and they do not create resources."""

from __future__ import annotations

from src.data_loader import PROJECT_ROOT
from src.predict import MODEL_PATH


def test_model_artifact_lives_inside_the_repository():
    assert MODEL_PATH == PROJECT_ROOT / "models" / "inference_pipeline.joblib"
    assert MODEL_PATH.is_file()
    assert MODEL_PATH.stat().st_size > 0
    assert (PROJECT_ROOT / "models" / "model_card.json").is_file()


def test_startup_files_use_the_repository_app():
    script = (PROJECT_ROOT / "scripts" / "start_streamlit.ps1").read_text(encoding="utf-8")
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    prep = (PROJECT_ROOT / "reports" / "aws_deployment_prep.md").read_text(encoding="utf-8")
    assert "streamlit run" in script
    assert "app\\app.py" in script
    assert "streamlit" in requirements
    assert "python -m streamlit run app/app.py" in prep
    assert "SageMaker" in prep


def test_env_example_and_gitignore_keep_secrets_out_of_git():
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    ignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    lowered = example.lower()
    assert "akia" not in lowered
    assert "aws_secret_access_key" not in lowered
    assert "aws_session_token" not in lowered
    assert "STREAMLIT_SERVER_PORT=8501" in example
    assert ".env" in ignore
    assert "!.env.example" in ignore
    assert "secrets." not in workflow
