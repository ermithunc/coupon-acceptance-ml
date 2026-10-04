"""Phase 20 checks. They do not call AWS and they do not create resources."""

from __future__ import annotations

import json

import joblib

from src.data_loader import PROJECT_ROOT
from src.predict import MODEL_PATH

PROD_REQUIREMENTS = (
    "pandas==3.0.6",
    "numpy==2.5.3",
    "scipy==1.18.1",
    "scikit-learn==1.9.1",
    "joblib==1.6.0",
    "streamlit==1.64.0",
    "xgboost==3.4.1",
)
DEV_ONLY = ("jupyter", "matplotlib", "seaborn", "shap", "pytest", "pyyaml")


def _requirement_names(text: str) -> list[str]:
    names: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name = line.split("[", 1)[0].split("=", 1)[0].split(">", 1)[0].split("<", 1)[0]
        names.append(name.strip().lower())
    return names


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


def test_production_requirements_are_the_inference_pins():
    path = PROJECT_ROOT / "requirements-prod.txt"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    names = _requirement_names(text)
    assert names == [line.split("==", 1)[0] for line in PROD_REQUIREMENTS]
    for line in PROD_REQUIREMENTS:
        assert line in text
    assert "xgboost>=" not in text
    assert ">=" not in "\n".join(
        raw.split("#", 1)[0] for raw in text.splitlines()
    )
    for name in DEV_ONLY:
        assert name not in names
    assert "nvidia-nccl-cu13" not in names


def test_deployment_installs_production_requirements_without_nccl():
    script = (PROJECT_ROOT / "scripts" / "install_inference.sh").read_text(encoding="utf-8")
    prep = (PROJECT_ROOT / "reports" / "aws_deployment_prep.md").read_text(encoding="utf-8")
    dev = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "requirements-prod.txt" in script
    assert "requirements-prod.txt" in prep
    assert "python3.13 -m pip install -r" in script
    assert "python3.13 -m pip install --no-deps" in script
    assert "xgboost>=" not in script
    assert "pip install pandas" not in script
    for name in DEV_ONLY:
        assert name not in script.lower()
    assert "jupyter" in dev
    assert "pytest" in dev
    assert "shap" in dev
    assert "python -m streamlit run app/app.py" in prep
    assert "python -m streamlit run app/app.py" in script


def test_saved_booster_matches_the_production_xgboost_pin():
    pipeline = joblib.load(MODEL_PATH)
    config = json.loads(pipeline.named_steps["model"].get_booster().save_config())
    assert config["version"] == [3, 4, 1]
    assert pipeline.named_steps["model"].get_params()["tree_method"] == "hist"
