Building a collection decision engine.

storage-intelligence/
│
├── data/
│   ├── raw/                  # Original downloaded data, never touch it
│   └── processed/            # Your cleaned, feature-engineered data
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feat_engineer.ipynb
│   ├── 03_classifier.ipynb
│   ├── 04_survival_analysis.ipynb
│   ├── 05_bayesian_engine.ipynb
│   ├── 06_uplift_model.ipynb
│   └── 07_decision_layer.ipynb
│
├── notes/                     # Decision log per notebook (see notes/<notebook>.md)
│   ├── 01_eda.md
│   ├── 02_feat_engineer.md
│   └── 03_classifier.md
│
├── src/
│   ├── features.py            # Reusable feature engineering functions
│   ├── models.py               # Model training/loading logic
│   └── decision.py             # Decision layer logic
│
├── api/
│   └── main.py                 # FastAPI endpoint
│
├── requirements.txt
├── README.md
└── .gitignore

## How this file works

`planning.md` holds the intent - what we plan to do next, and why - before it's implemented. Once a notebook's work is actually done, the detailed reasoning, edge cases, bugs found, and results move into that notebook's file under `notes/`. This file should stay short enough to scan in one read.

## Where things stand

- `01_eda.ipynb`: preprocessing complete (column drops, target/split creation, imputation, leakage audit, train/val/test split). See `notes/01_eda.md`.
- `02_feat_engineer.ipynb`: complete, including a Round 2 test of the 5 deferred feature ideas below - all landed within noise-level of baseline. See `notes/02_feat_engineer.md`.
- `03_classifier.ipynb`: baseline logistic regression + Random Forest comparison, two rounds of hyperparameter tuning done (see "Next up" plan below, now executed) - converged around ~0.72-0.723 ROC-AUC with diminishing returns. See `notes/03_classifier.md`.

## Original plan for 03_classifier.ipynb (executed)

**Next up:** train a Random Forest (or XGBoost) on the same train/val split as the logistic regression baseline, for a direct ROC-AUC/F1 comparison. Trees should be naturally more robust to two issues found in the linear model (sentinel-999 scaling distortion, `addr_state` one-hot sparsity/small-sample inflation) without extra preprocessing - this tells us whether those issues actually cost the linear model anything, and whether tree-based is worth pursuing further for this classifier.

**After that**, depending on results:
- Tune the decision threshold (ROC/precision-recall curve) instead of using the default 0.5 - directly relevant to the eventual collections decision.
- Consider addressing `addr_state` sparsity (regional grouping, or regularized/target encoding) if it's shown to hurt performance.
- Revisit the deferred feature ideas (sub_grade numeric encoding as its own feature, `revol_util` x `dti` interaction, high-utilization flag, active-account ratio, purpose grouping) guided by feature importances from whichever model wins.
- Once a baseline classifier is solid, move to `04_survival_analysis.ipynb`.

**Outcome:** Random Forest tied the logistic regression baseline. Two rounds of hyperparameter tuning converged around 0.72-0.723 ROC-AUC with diminishing returns. All 5 deferred features were tested (in `02_feat_engineer.ipynb` Round 2) and also landed within noise-level of baseline. Decision threshold tuning and `addr_state` sparsity were not pursued - see "Current plan" below for why.

## Current plan

Both hyperparameter tuning and feature engineering plateaued at ~0.72 ROC-AUC independently - two separate investigations pointing the same direction. This looks like a genuine ceiling for origination-time-only prediction with this feature set, not an artifact of under-tuning or under-engineering.

**Next up:** move to `04_survival_analysis.ipynb`. The plateau suggests the next real gains come from mid-loan-life/behavioral data (hardship/settlement events, time-to-default), which is exactly what survival analysis brings in - not from further tuning or feature tweaks on the initial classifier.

**Parked for later** (not urgent, revisit only if it becomes relevant):
- Decision threshold tuning (ROC/precision-recall curve) instead of the default 0.5 - relevant once closer to an actual collections deployment decision.
- `addr_state` one-hot sparsity/small-sample coefficient inflation (flagged in the logistic regression) - never confirmed to actually hurt performance, so not addressed.
- A more rigorous hyperparameter search (`HalvingRandomSearchCV`, Bayesian optimization) if tuning ever becomes worth revisiting - the current search was small (12 samples of 100+ combinations each round) and diminishing but not exhaustively proven to have hit a hard ceiling.

## Longer-term / deferred work

- **Time-based train/test split** by `issue_d` (train on earlier-issued loans, test on later ones) - more realistic for eventual deployment given loans span 2007-2018 with different economic conditions. Deferred until the simple model is working; revisit alongside survival analysis/uplift/deployment.
- **Hardship/settlement features** (`hardship_flag`, `debt_settlement_flag`, and the granular hardship/settlement fields dropped in `01_eda.ipynb`): excluded from the initial classifier as leaky, but relevant for survival analysis and uplift modeling later.
- **payment_ratio / loan_ratio**-style features (using mid-loan payment history like `last_pymt_amnt`, `total_rec_prncp`): need data that doesn't exist at origination - deferred to a future behavioral/collections-scoring model that operates mid-loan-life, not this initial classifier.
- Any other leakage-sensitive decisions made for simplicity in the initial classifier should be revisited once survival analysis, uplift modeling, and deployment stages begin.

## Metric/model reasoning (still relevant going forward)

**Class imbalance:** 1:3.75 (Class 0: 78.96%, Class 1: 21.04%) - in the range where `class_weight='balanced'` is sufficient (no resampling/SMOTE needed):

| Imbalance Ratio | Treatment |
|---|---|
| 1:2 to 1:5 | `class_weight='balanced'` is enough. |
| 1:5 to 1:20 | Start considering mild resampling — undersample majority, or try SMOTE |
| 1:20 to 1:100 | SMOTE or other oversampling becomes necessary |
| 1:100+ (fraud, rare disease) | Aggressive techniques — SMOTE + undersampling combined, anomaly detection framing, or treat it as a different problem entirely |

**Primary metric: ROC-AUC** - threshold-independent, matters because the actual collections decision threshold isn't decided yet. **Secondary: F1-score.**
