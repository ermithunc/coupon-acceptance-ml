# AWS deployment preparation

No AWS resources were created. `aws sts get-caller-identity` was not run because the AWS CLI is not installed on this machine (`aws` is not on PATH, and `C:\Program Files\Amazon\AWSCLIV2\aws.exe` is absent).

## Checklist

| Item | State |
| --- | --- |
| requirements.txt | Present. Includes `streamlit` and `xgboost`. |
| Startup command | From the repository root: `python -m streamlit run app/app.py`. Windows helper: `scripts/start_streamlit.ps1`. |
| Model artifact | `models/inference_pipeline.joblib`, loaded from the project root recorded in code. `models/model_card.json` is the claim record. |
| Environment variables | `.env.example` sets Streamlit port, address, and headless mode. No secret values. |
| Relative paths | The app and `load_pipeline` resolve the repository from `__file__`, so the process does not depend on the shell's current directory. |
| Secret handling | `.gitignore` ignores `.env`, `*.pem`, `*.key`, and `credentials.json`. The CI workflow does not read GitHub secrets. |
| Paid infrastructure | None. No SageMaker, ECS, Kubernetes, or EC2 was requested or created. |

## Startup

Linux or the future host, with the working directory set to the repository root:

```bash
python -m streamlit run app/app.py
```

Streamlit reads `STREAMLIT_SERVER_PORT`, `STREAMLIT_SERVER_ADDRESS`, and `STREAMLIT_SERVER_HEADLESS` from the environment. The example file uses port 8501, address `0.0.0.0`, and headless mode.

## What has to happen before Phase 21

1. Install AWS CLI v2.
2. Authenticate on this machine with `aws configure` or AWS SSO. Leave the keys in the local AWS credential store.
3. Run `aws sts get-caller-identity` and confirm the account before any resource is created.
4. Do not paste access keys, secret keys, or session tokens into chat, source files, or GitHub.
