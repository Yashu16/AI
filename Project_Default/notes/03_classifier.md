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

## Next: tree-based comparison

Training a Random Forest/XGBoost on the same train/val split for direct ROC-AUC comparison. Trees split on thresholds rather than magnitude, so they should be naturally more robust to both issues found above (sentinel-999 scaling distortion, `addr_state` one-hot sparsity/small-sample inflation) without needing the same preprocessing fixes - a good test of whether those issues were really costing the linear model anything, and whether a tree-based approach is worth pursuing further.
