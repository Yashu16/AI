# 03_classifier.ipynb - Decision Log

## Baseline model choice

Model: Logistic Regression, `class_weight='balanced'` (per `01_eda.ipynb` decision, 1:3.75 imbalance). Primary metric ROC-AUC (threshold-independent - deployment threshold for collections isn't decided yet), secondary F1.

## Preprocessing

`X_train` has 76 cols going in: 60 float64, 11 str, 4 int64 missing-flags, 1 int32 (`credit_history_length`).

- **`issue_d`, `earliest_cr_line`**: still present as raw date strings even though only used to derive `credit_history_length` in `02_feat_engineer.ipynb`. Dropped both - signal already captured in `credit_history_length`, and as raw strings they'd one-hot encode into hundreds of near-useless sparse columns.
- **`grade`, `sub_grade`, `emp_length`**: ordinal (known ranking). `OrdinalEncoder`, not one-hot - preserves ranking as a single numeric column, more efficient for a linear model than re-learning the order across many separate one-hot coefficients.
- **`term`, `home_ownership`, `verification_status`, `purpose`, `addr_state`, `application_type`**: genuinely unordered categories. One-hot encoded.
- **All numeric columns**: `StandardScaler` (logistic regression's optimizer is scale-sensitive).
- All fit on `X_train` only via a single sklearn `Pipeline` + `ColumnTransformer`, applied to `X_val`/`X_test` without refitting.

## Bug: sentinel-value distortion in coefficients

First coefficient peek was dominated by `mths_since_recent_inq`/`mths_since_recent_inq_missing_flag` and `mths_since_last_delinq`/`mths_since_last_delinq_missing_flag` - not real credit-risk signal, but an artifact of the `01_eda.ipynb` Step 3 sentinel fill (999).

Confirmed: real (non-missing) values for these columns top out far below 999 (`mths_since_recent_inq` real max = 25, `mths_since_last_delinq` real max = 202, `num_tl_120dpd_2m` real max = 6, `mo_sin_old_il_acct` real max = 724) - the sentinel is 5x-800x larger than any genuine value, so after scaling it becomes an extreme outlier that dominates a linear model's coefficients, even though the `_missing_flag` already captures "was this missing" cleanly.

**Decision:** for the linear model, keep the `_missing_flag` columns but re-impute the 4 sentinel-filled columns with a milder value (max non-missing value + 1) instead of dropping the raw column outright - preserves real information for non-missing rows without injecting an arbitrary huge outlier.

This is an empirical, not purely reasoned, choice - there's no way to know in advance whether milder-sentinel or drop-raw-column performs better, so compare ROC-AUC/F1 on `X_val`. For tree-based models, try both the original 999-sentinel version and the milder version, since trees split on thresholds rather than magnitude and may not be affected the same way.

*(Professional practice for this kind of comparison is usually formal experiment tracking - MLflow, Weights & Biases, DVC; for this project's scale, tracking results directly in these notes is a reasonable substitute.)*

## addr_state coefficient noise

Re-ran the coefficient peek after the mild-sentinel fix: sentinel distortion gone. New top-15 led by `purpose_wedding`/`purpose_small_business`/`term_36_months` (domain-sensible) but also 11 `addr_state` one-hot columns.

Checked sample sizes: the states with the largest coefficients (ME 1,472, VT 1,890, DC 2,469, NE 2,621, WV 3,473) are the smallest-sample states in the dataset (vs CA 140,777, TX 79,086) - classic small-sample coefficient inflation from one-hot encoding a ~50-category column with very uneven group sizes. CA/TX/NY/FL/IL (largest states) don't appear in the top 15 at all.

Not necessarily wrong, but less trustworthy than a coefficient backed by more data. Flagged as a possible improvement (grouping states into regions, or regularized/target encoding) but deferred until we see whether it's actually hurting predictive performance.

## Baseline logistic regression - validation results

ROC-AUC: 0.7178, F1 (default class): 0.4458

| | precision | recall | f1-score | support |
|---|---|---|---|---|
| No Default | 0.88 | 0.66 | 0.76 | 163415 |
| Default | 0.34 | 0.65 | 0.45 | 43165 |

Confusion matrix: `[[108629, 54786], [15067, 28098]]`

**Interpretation:** ROC-AUC of 0.72 is a modest-but-real signal - meaningfully better than random (0.5) but well short of a strong discriminator (0.85+). `class_weight='balanced'` is visibly doing its job: recall on the Default class is 0.65 (catching most actual defaulters) at the cost of precision (0.34 - a lot of false alarms, 54,786 non-defaulters flagged as risky). This tradeoff is intentional and expected given the imbalance correction, and is exactly why threshold tuning will matter later for the actual collections decision - this default 0.5 threshold is not necessarily where you'd want to operate.

This is a first-pass baseline, not a final model - reasonable starting point given: (a) no hyperparameter tuning done yet, (b) `addr_state` one-hot noise identified but not yet addressed, (c) only 5 engineered features added on top of raw columns, (d) tree-based model not yet tried for comparison.

## Tree-based comparison: Random Forest

Trained `RandomForestClassifier` (`n_estimators=200`, `max_depth=12`, `class_weight='balanced'`) on the same train/val split - original 999 sentinel (not the mild-sentinel version), ordinal/label-encoded categoricals (no one-hot), no scaling.

**Result (validation):**

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| ROC-AUC | 0.7178 | 0.7167 |
| F1 (Default) | 0.4458 | 0.4438 |
| Precision (Default) | 0.34 | 0.33 |
| Recall (Default) | 0.65 | 0.67 |

**Interpretation:** essentially a tie - the ~0.001 ROC-AUC gap is noise-level, not a meaningful difference. This is informative on its own:

- The sentinel-999 and `addr_state` sparsity issues we found in the linear model apparently weren't costing much in aggregate predictive power - the Random Forest handled both "for free" (no re-imputation, no one-hot) and landed in the same place. Consistent with the idea that those were coefficient-interpretability problems more than performance problems.
- A `max_depth=12`, untuned Random Forest matching a linear model suggests either (a) the signal in this feature set is close to linear/simple - i.e. most of the discriminative power comes from a handful of strong features (grade, FICO, DTI, utilization) that a linear model captures just as well, or (b) neither model has enough hyperparameter tuning yet to show a real gap, or (c) both are running into the same ceiling: 71 features engineered so far may not carry enough signal to push ROC-AUC much past ~0.72 regardless of model family.
- Given the tie, model family isn't the bottleneck right now - future effort is likely better spent on hyperparameter tuning (both models are essentially untuned defaults) and additional feature engineering (the deferred ideas: sub_grade as numeric, interactions, purpose grouping) rather than trying more model architectures.

**Decision:** don't invest further in exhaustively comparing model families yet. Revisit tree-based vs linear only after feature engineering is pushed further and/or hyperparameters are tuned - a fair comparison needs both sides to be closer to their ceiling first.

## Hyperparameter tuning

Ran `RandomizedSearchCV` on the Random Forest, scored on ROC-AUC. First attempt (`n_iter=15`, `cv=3`, including `n_estimators=400` and `max_depth=None`) ran 100+ minutes without finishing - interrupted and scaled down to `n_iter=6`, `cv=2`, dropping the slowest param values (400 trees, unbounded depth). Search space: `n_estimators` [100, 200], `max_depth` [8, 12, 16], `min_samples_leaf` [5, 20, 50], `max_features` ['sqrt', 0.3].

**`max_features` note:** tried both `'sqrt'` (≈8 of 74 features per split - the traditional Random Forest default, maximizes tree diversity/decorrelation) and `0.3` (≈22 features per split - lets trees pick from stronger candidates at the cost of more inter-tree correlation). Genuinely unclear in advance which suits this dataset (some strong features like grade/FICO, many weaker ones like individual `addr_state` dummies), so left it to the search rather than guessing.

**Result:**

| Model | ROC-AUC (val) | F1 (Default, val) |
|---|---|---|
| Logistic Regression (baseline) | 0.7178 | 0.4458 |
| Random Forest (untuned) | 0.7167 | 0.4438 |
| Random Forest (tuned) | 0.7215 | 0.4490 |

Best params: `n_estimators=200, min_samples_leaf=50, max_features=0.3, max_depth=16`. Best CV ROC-AUC: 0.7189.

**Interpretation:** tuning helped, but only modestly (+0.005 over untuned RF, +0.004 over logistic regression baseline) - not the kind of jump that suggests model capacity was the real bottleneck. This is consistent with (and reinforces) the feature-set-ceiling hypothesis: even with better hyperparameters, ROC-AUC is still capped around ~0.72, not jumping toward 0.80+. 

**v2 result:** val ROC-AUC 0.7227 (vs v1's 0.7215) - a gain of only +0.0012, and F1 barely moved (0.4490 -> 0.4491). Notably, `min_samples_leaf=50` and `max_features=0.3` landed on the *exact same* values in both v1 and v2, independently - a real signal these two are near their true optimum, not truncated by the search range. Only `n_estimators` (200->400) and `max_depth` (16->None) pushed further, but with `min_samples_leaf=50` already forcing large leaves, an unbounded `max_depth` can't overfit much further than a capped one - the leaf-size floor is doing the real regularization work, which is why depth becoming unbounded barely moved the score. More trees (200->400) showing almost no gain is the classic diminishing-returns curve of ensemble size.

**Decision:** hyperparameter tuning appears to have hit diminishing returns for this Random Forest at ~0.72-0.723 ROC-AUC. Move effort to feature engineering next.

**Caveat on how much to trust this:** `RandomizedSearchCV` with `n_iter=6` only tries 6 of the 108 possible combinations in the v2 grid (3x4x3x3) - it reports the best *of the 6 sampled*, not the best possible. Two independent rounds landing on the same `min_samples_leaf=50`/`max_features=0.3` is suggestive (less likely to be pure coincidence than one round alone), but with only 12 total samples out of 108+ combinations, this is weak evidence, not a rigorous guarantee that a fuller search wouldn't find something meaningfully better. "Confirmed" was too strong a word for what a search this size can establish.

A more rigorous option without going fully exhaustive: `HalvingRandomSearchCV` (successive halving - tries many candidates on a small data subset first, only lets survivors run on more data, so more combinations can be explored for similar compute) or Bayesian optimization (Optuna, scikit-optimize's `BayesSearchCV` - each trial informed by previous ones instead of random). Not pursued for now given the small, consistently diminishing gains already seen (0.7167 -> 0.7215 -> 0.7227) - the trend itself is informative even if the exact ceiling isn't pinned down exactly. Revisit with a more rigorous search only if feature engineering stalls and tuning becomes worth another look.

**Correction:** re-examined the tuned RF's best params - `n_estimators=200`, `max_depth=16`, `min_samples_leaf=50`, `max_features=0.3` all sat at or near the edge of the tested range. That's a sign the search may have been cut off before finding the true optimum, not that we found a real peak - the "tuning barely helps" conclusion above was premature given the search only tried 6 of ~36 combinations. Running a second, wider search (still time-boxed at `n_iter=6`, `cv=2`) with ranges shifted past the v1 edges - see experiment log below for tracking across runs.

## Experiment log

Keeping a running table here instead of a formal tracker (MLflow/W&B) given the project's current scale - see the earlier note on professional practice for this kind of comparison.

| Run | Model | Key params | CV ROC-AUC | Val ROC-AUC | Val F1 (Default) |
|---|---|---|---|---|---|
| 1 | Logistic Regression | `class_weight='balanced'`, default `C=1.0`, mild-sentinel imputation | — | 0.7178 | 0.4458 |
| 2 | Random Forest (untuned) | `n_estimators=200, max_depth=12, class_weight='balanced'`, original 999 sentinel | — | 0.7167 | 0.4438 |
| 3 | Random Forest (tuned v1) | `n_estimators=200, max_depth=16, min_samples_leaf=50, max_features=0.3` | 0.7189 | 0.7215 | 0.4490 |
| 4 | Random Forest (tuned v2) | `n_estimators=400, max_depth=None, min_samples_leaf=50, max_features=0.3` | 0.7196 | 0.7227 | 0.4491 |
