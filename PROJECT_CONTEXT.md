# Project context

Handoff summary as of 2026-10-01. Workspace: `D:\HACKATHON-ACHIEVERS`.

Checked against `Coupon_Acceptance_Hackathon_CURSOR_Meta_Prompt.md`, the requirements document `Coupon_Acceptance_ML_Hackathon_Requirements_and_Implementation_Plan.docx`, `git log`, `git branch -a`, and the files on `main`.

## Mission

Predict whether a customer will accept a recommended restaurant or bar coupon.

- `Y = 1` accepted
- `Y = 0` not accepted

Official files, inspected and matching the stated sizes:

- Train: 10,147 rows, 27 columns, including `Y`
- Test: 2,537 rows, 26 columns, no `Y`
- `sample_submission.csv`: `customer_id`, `Y`, same id order as the test file

Local raw CSVs live in `data/raw/` (`train.csv`, `test.csv`, `sample_submission.csv`) and are gitignored. The original drop folder `Dataset/Datasets/` is still present, also gitignored, and is only a fallback. `src/data_loader.py` reads `data/raw/` first. Do not modify the raw files. Do not use the test file for fitting, feature learning, tuning, threshold choice, or model selection.

Column spelling `passanger` is intentional. `customer_id` is an identifier. `car` is kept and encoded as its own missing level (99.17% missing in train). `toCoupon_GEQ5min` is constant 1 and is dropped from the model matrix. `direction_opp` is the exact complement of `direction_same` and is dropped.

## Architecture

Training stays on this machine. GitHub is source control. AWS is reserved for a later lightweight Streamlit deployment.

Repository: https://github.com/ermithunc/coupon-acceptance-ml  
Remote: `origin`, branch `main`, public. Latest pushed commit: `8e67e17` (`docs: record GitHub remote`).

`git branch -a` shows only `main` and `origin/main`. There are no feature branches and no pull requests.

Python: 3.13.15 in `.venv`. `python` is not on PATH. Use `.venv\Scripts\python.exe` or `py -3.13`.

## What is done

Modeling phases 2–17 have their artifacts and one commit each. Phase 1’s GitHub-remote exit criterion was finished later, in `8e67e17`, after Phase 17. See the divergence section below.

| Phase | Commit | Outcome |
| --- | --- | --- |
| 1 | `aa96ede` | Project layout, venv, local git. Remote was not created in this commit. |
| 2 | `1f0f970` | `src/data_loader.py`, data dictionary. Sizes match the official statement. |
| 3 | `a91ef2e` | Data-quality report. Nothing auto-cleaned. 63 feature-duplicate pairs in train; 15 have mixed `Y`. |
| 4 | `d6caef5` | Training-set EDA. Overall acceptance rate 56.84% (5,768 / 4,379). Notebook `notebooks/02_eda.ipynb`. |
| 5 | `85e4088` | sklearn preprocessing pipeline. One-hot nominals, scale `temperature` only. |
| 6 | `1d4bb82` | Engineered features beat the plain pipeline on a logistic-regression probe: ROC-AUC 0.7590 vs 0.7360. |
| 7 | `1d8a85a` | Shared protocol: engineered pipeline inside each fold, stratified 5-fold, `random_state=42`, threshold 0.5. |
| 8 | `b0bd48f` | Untuned logistic regression. |
| 9 | `2394ef2` | Untuned decision tree. |
| 10 | `a4188db` | Untuned random forest. |
| 11 | `56c0a22` | Untuned XGBoost. |
| 12 | `3e92146` | Comparison. Shortlist: XGBoost and random forest (within 0.02 ROC-AUC). `notebooks/05_model_comparison.ipynb`. |
| 13 | `bedf99e` | Bounded `RandomizedSearchCV`: 8 draws, 3 folds, scoring ROC-AUC, then a fresh 5-fold rescore. |
| 14 | `2e19f67` | Selected model: tuned XGBoost. `reports/final_model_decision.md`. |
| 15 | `f041ab0` | SHAP on a 400-row training sample. Association language only. |
| 16 | `c11b98e` | `models/inference_pipeline.joblib` and `submission.csv`. |
| 17 | `cbbe984` | Streamlit app at `app/app.py`. |
| — | `8e67e17` | GitHub remote recorded and the existing `main` history pushed. This is the late Phase 1 exit, not a new modeling phase. |

`PROJECT_CONTEXT.md` records this handoff. It is committed with the layout checkpoint that follows `8e67e17`.

## Selected model

Tuned XGBoost: 200 trees, `max_depth=6`, `learning_rate=0.1`, `min_child_weight=1`, `subsample=0.8`.

Performance claim, from 5-fold cross-validation on the training file (`reports/experiments/xgboost_tuned.json`, copied into `models/model_card.json`):

- ROC-AUC 0.8381 ± 0.0062
- PR-AUC 0.8596
- F1 0.8017
- Precision 0.7715
- Recall 0.8344
- Accuracy 0.7654
- Brier 0.1606

The saved joblib was refit on all 10,147 training rows after that score was locked. Full-refit training accuracy is 0.8900. That number is not the performance claim.

`submission.csv` has 2,537 rows, ids aligned with the test file, no missing values. Predicted 0: 1,021. Predicted 1: 1,516. Threshold 0.5.

Untuned and tuned comparison numbers used for selection are in `reports/final_model_decision.md` and `reports/model_comparison.md`. Global SHAP associations are in `reports/shap_explanations.md` (largest mean absolute SHAP: `coupon_venue_freq_ord`, then 1-day expiration).

## Demo

From the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

The app loads the saved pipeline. It does not retrain. Default form is a 21-year-old man, with friends, Coffee House coupon, sunny, 6PM, no urgent place. A Streamlit AppTest of that form returned 84.6% and class Accept. That AppTest is not a committed test.

The in-IDE browser tab stayed blank after a connection reset. The predict path was verified with Streamlit's AppTest runner.

## Where the work left the meta prompt

The model path (real CSVs, leakage rules, four model families, shared 5-fold protocol, evidence-based selection, SHAP wording, local training) follows the meta prompt. The git process does not.

### Branching

The meta prompt never names a branch-per-phase scheme such as `phase/08-logistic`. What it does require:

- Phase 1 exit criteria include “GitHub remote works” before that phase stops (`Coupon_Acceptance_Hackathon_CURSOR_Meta_Prompt.md`, Phase 1).
- After every phase: implement, test, document, git checkpoint, then stop. Section 5 and section 38 say one phase at a time and “Never execute multiple phases at once.”
- Phase 19 requires GitHub Actions on `push` and on `pull request`.

What the repository has:

- One branch, `main`, from the first commit through `8e67e17`.
- No feature branches, no pull requests, and no merge commits.
- The GitHub remote did not exist during phases 2–17. `origin` was added only when `8e67e17` was pushed, so the whole history landed on `main` in one push.
- `.github/workflows/` contains `.gitkeep` only. `ci.yml` is still Phase 19.

A pull-request workflow cannot be reconstructed from this history without new branches. Do not rewrite the existing commits. If later phases need a pull request, start a new branch from current `main` and open the PR from there. Leave phases 2–17 as the linear `main` history they already are.

The requirements document’s “Immediate Execution Plan” says to create the GitHub repo first, then push again after Streamlit when CI is configured. The meta prompt is stricter: the remote is a Phase 1 exit criterion. The late remote follows the requirements timing and misses the meta-prompt timing.

### Other process gaps

- Phases were batched when asked (“next 2”, “next 4”, “next 5”, “continue”). Each phase still has its own commit. The stop-after-one-phase rule was not followed on those turns.
- `PROJECT_STATUS.md` still lists Last Git Commit as `feat: add coupon acceptance streamlit demo`. HEAD is `8e67e17` (`docs: record GitHub remote`). The GitHub section in that file is the accurate remote record. Current phase 17 and next action Phase 18 are still right.
- `README.md` is still the Phase 1 stub. It says modeling, Streamlit, CI, and AWS have not started. Phase 23 is the designated README rewrite, so this file was left stale on purpose under the phase plan. Anyone reading only the README will get the wrong status.
- Phase 1 continued while `gh` was not logged in. Section 2 says to stop the affected phase when authentication is missing. The remote was completed after the user finished `gh auth login` as `ermithunc`.

### Layout check against the meta prompt tree

Checked on disk and in git. These paths exist:

- `data/raw/` and `data/processed/`
- `notebooks/`, `src/`, `models/`, `reports/figures/`
- `app/`, `tests/`, `.github/workflows/`
- `requirements.txt`, `README.md`, `PROJECT_STATUS.md`, `.gitignore`

`data/raw/` now holds the local official CSVs. `.gitignore` excludes `data/raw/*.csv`, `data/processed/*` except `.gitkeep`, and `Dataset/`. Those CSVs are not part of the GitHub tree. `.github/workflows/ci.yml` is still Phase 19; the folder exists with `.gitkeep` only.

This layout checkpoint is committed on `docs/repository-layout` and opened as a pull request, instead of another direct commit to `main`.

This layout checkpoint is committed on `docs/repository-layout` and opened as a pull request, instead of another direct commit to `main`.

- The meta prompt asks for two notebooks: `notebooks/02_eda.ipynb` and `notebooks/05_model_comparison.ipynb`. Both exist. The requirements document also lists `01_data_understanding.ipynb`, `03_feature_engineering.ipynb`, `04_baseline_models.ipynb`, and `06_final_model.ipynb`. Those four were not required by the meta prompt and were not created.
- The requirements document uses 15 phases and folds tree, forest, and XGBoost into one comparison phase. The meta prompt expands that into phases 7–14. The commits follow the 25-phase meta prompt. That split is intentional.
- Requirements Phase 7 asks to inspect logistic coefficients. The meta prompt’s Phase 8 asks for the shared metrics only. No coefficient table was saved.
- Phase 18’s test list is still open: model loading, single-row inference, batch inference, missing values, invalid categories, prediction type, probability range, feature schema, and a committed Streamlit smoke test. `tests/test_predict.py` checks the saved submission contract. `tests/test_demo_text.py` checks the claim sentence and profile text. Neither file is the Phase 18 suite.

### Extra work that was requested and is in the tree

- `src/model_card.py`, `models/model_card.json`, and `src/demo_text.py` separate the 5-fold ROC-AUC claim from the full-training refit. That fix was requested after Phase 17 and is included in `8e67e17` together with the remote note.
- `app/app.py` shows the profile sentence, the claim sentence, the stored untuned comparison table, and a short project path. Phase 17 only requires the form, the saved pipeline, the probability, and the class. The comparison table is stored validation evidence. The live row is scored only by the tuned XGBoost pipeline.
- A judge-demo canvas was drafted outside this git repo. It is not part of `main`.

## Rules for the next agent

- One phase at a time unless the user explicitly asks for more.
- Do not invent metrics, columns, or categories.
- Do not fit anything on `test.csv`.
- Do not drop `passanger` spelling or silently drop `car`.
- SHAP text must say the model associated a feature with higher or lower predicted probability. Do not say a feature causes acceptance.
- Do not ask the user to paste tokens or AWS keys.
- AWS CLI is not installed. Do not create paid infrastructure unless the deployment phase is reached and the user approves it.
- Do not rewrite `main` to invent historical branches. New phase work that should be reviewed as a pull request starts from a new branch.

## Phase 18

`tests/test_inference.py` loads the saved pipeline and does not refit. `pytest tests/test_inference.py` passed 7 tests. The default form row scores 0.846452 (84.6%, class Accept). Scoring all 2,537 official test rows reproduces `submission.csv` at threshold 0.5. Missing `car`, `Bar`, and `CoffeeHouse` still return a probability in `[0, 1]`. Unknown categories `Not A Real Coupon` and `Mars` do not raise, because one-hot encoding uses `handle_unknown="ignore"`. The Streamlit AppTest metric matches that direct score. Details are in `reports/inference_testing.md`.

## Still open

- Phase 19 — GitHub Actions CI on push and pull request (this is the first phase that needs a branch other than a direct commit to `main`, if the pull-request trigger is going to be real)
- Phase 20–22 — AWS preparation, lightweight deploy, cost notes
- Phase 23–25 — README, presentation material, final audit

GitHub login is done as `ermithunc`. The remaining deployment blocker is the missing AWS CLI, and that is only needed from Phase 20.
