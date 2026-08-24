# 02_feat_engineer.ipynb - Decision Log

## Scope clarification

`payment_ratio` (`last_pymt_amnt`/`installment`) and `loan_ratio` (`total_rec_prncp`/`loan_amnt`), originally sketched in `01_eda.ipynb`, both need mid-loan payment history that doesn't exist at origination - and both reference columns that are either leaky (`total_rec_prncp`) or misspelled/no longer present (`last_pymt_amnt`). The classifier predicts default risk using only what's known when the loan is issued (Bucket 4: credit profile, loan terms, borrower financials), so there's no "payment behavior so far" to feature-engineer from at this stage.

Decision: build features from what's actually available in `X_train` (origination-time columns) instead. `payment_ratio`/`loan_ratio`-style features are deferred to a future behavioral/collections-scoring model that operates mid-loan-life, not this initial classifier.

## Initial feature set (5)

All built from origination-time columns already cleared in the `01_eda.ipynb` leakage audit:

1. `installment_to_income` = `installment / (annual_inc / 12)` - loan payment as share of monthly income
2. `loan_to_income` = `loan_amnt / annual_inc` - loan size relative to income
3. `credit_history_length` = months between `earliest_cr_line` and `issue_d` (both parsed via `pd.to_datetime`)
4. `avail_credit_ratio` = `bc_open_to_buy / total_rev_hi_lim` - revolving credit headroom relative to limit
5. `fico_avg` = `(fico_range_low + fico_range_high) / 2` - single combined FICO score

**Decision on scope:** stick with these 5 for the first pass rather than pre-building more (sub_grade numeric encoding, `revol_util` x `dti` interaction, high-utilization flag, active-account ratio, purpose grouping were considered). Train a baseline classifier first, then use feature importances/coefficients to decide what's actually worth adding - avoids over-engineering before seeing what the model needs.

## Edge cases found (via `.describe()` on `X_train`)

- **`avail_credit_ratio`**: `total_rev_hi_lim == 0` for 48,155 rows (real zeros in raw data, not an artifact of Step 3's zero-fill) - division produces `inf`, which poisons `mean()`/`max()` for the whole column and would break most non-tree models outright.
- **`installment_to_income`, `loan_to_income`**: a handful of rows have implausibly low `annual_inc` (as low as $20-$100), producing extreme ratios (up to 282x/820x) - not `inf`, but could dominate models sensitive to feature magnitude/scale.
- **`credit_history_length`**: long tail of very old `earliest_cr_line` dates (155 rows before 1960, 912 rows > 50 years) - smooth continuous tail, not a spike, likely a mix of genuine long-lived accounts and data-entry errors that can't be distinguished from the data alone. One row (`earliest_cr_line` Apr-1934) coincidentally hit exactly 999, colliding with the unrelated sentinel value used in `01_eda.ipynb` Step 3 - coincidence, not a real link, but worth noting as a landmine if these columns are ever compared directly.

**Decision:** clip all three (rather than drop rows or leave as-is) - preserves every row's other features, treats the extreme tail as "very high/long" rather than needing to verify or fix each value. Clip bounds computed on `X_train` only (99th percentile) and the same bounds applied to `X_val`/`X_test` (train-only-fit, avoids leakage from val/test statistics).

## Bug caught during the fix

Initially used `pd.NA` to replace `inf` in `avail_credit_ratio`, which silently cast the column to `object` dtype (not `float64`) even after `fillna(0)` - would have broken model training later since most classifiers need numeric dtypes. Fixed by using `np.nan` instead, which stays within `float64`.

**Lesson: always check `.dtypes` after a `replace()`/`fillna()` involving `pd.NA`, not just `.describe()` - object dtype can look fine in a numeric summary until you check.**

## Result

- `installment_to_income` clipped to max 0.201, `loan_to_income` to max 0.5, `credit_history_length` to max 478 months (~40 years) - bounds from `X_train`'s 99th percentile.
- `avail_credit_ratio`: 48,155 rows with `inf` converted to `NaN`, then filled with 0 (no revolving limit = no headroom) - full 964,040 count, range 0-1, all `float64`.

Saved to `data/processed/train_fe.parquet`, `val_fe.parquet`, `test_fe.parquet` for handoff to `03_classifier.ipynb`.

## Round 2: testing the deferred feature ideas

After the baseline classifier (logistic regression + Random Forest) plateaued around 0.72 ROC-AUC even after hyperparameter tuning (see `notes/03_classifier.md` experiment log), decided the feature set - not the model - is the likely bottleneck. Going back to the 5 ideas deferred in Round 1:

1. `sub_grade_numeric` - A1=1...G5=35, persisted as an actual feature column (currently only exists inside the classifier pipeline's ordinal encoder, not saved to the processed data)
2. `revol_util_x_dti` = `revol_util * dti` - interaction term, "high utilization AND high DTI" compounding risk
3. `high_utilization_flag` = `bc_util > 80` - binary flag for a "danger zone" nonlinearity a raw ratio might average out
4. `active_account_ratio` = `num_actv_rev_tl / total_acc` - proportion of credit history that's currently active/live
5. `purpose_grouped` - collapse the 14 `purpose` categories into fewer, better-populated buckets

**Decision on process:** add incrementally and test each one's ROC-AUC/F1 impact on the Random Forest before adding the next, rather than building all 5 and evaluating the combined effect - slower, but tells us which specific features actually help vs. hurt vs. do nothing, rather than an ambiguous combined delta.

**Quick-eval setup:** `quick_eval()` helper trains a Random Forest with the tuned params found in `03_classifier.ipynb` (`n_estimators=200` for speed rather than 400, `max_depth=None, min_samples_leaf=50, max_features=0.3, class_weight='balanced'`) and reports val ROC-AUC/F1. Note this baseline still includes the raw `issue_d`/`earliest_cr_line` string columns (ordinal-encoded) that `03_classifier.ipynb` drops separately - not perfectly apples-to-apples with that notebook's numbers, but fine for measuring the *relative* delta each new feature adds within this loop.

### Round 2 experiment log

| Step | Feature added | Val ROC-AUC | Val F1 | Delta vs. previous |
|---|---|---|---|---|
| 0 | Baseline (Round 1's 5 features) | 0.7224 | 0.4493 | — |
| 1 | + `sub_grade_numeric` (A1=1...G5=35) | 0.7226 | 0.4503 | ROC-AUC +0.0002, F1 +0.0010 |
| 2 | + `revol_util_x_dti` = `revol_util * dti` | 0.7226 | 0.4493 | ROC-AUC +0.0002 (vs baseline), -0.0001 (vs step 1); F1 essentially flat |

**Data quality note found while checking this feature:** `revol_util_x_dti` had a suspicious min of -78.2. Traced to a single row where `dti = -1.0` (out of 964,040) - `revol_util` itself has no negative values (min 0.0). A DTI of exactly -1.0 is almost certainly a raw-data sentinel/placeholder from Lending Club, not a genuine negative debt-to-income ratio. Affects only 1 row - immaterial to model performance, not worth a fix at this scale, but noted here in case it resurfaces (e.g. if a future feature divides by `dti` and this row produces an extreme/wrong-signed result).

| 3 | + `high_utilization_flag` (`bc_util > 80`, ~29.4% of rows flagged) | 0.7227 | 0.4495 | ROC-AUC +0.0001 (vs step 2), +0.0003 (vs baseline); F1 +0.0001/+0.0002 |
| 4 | + `active_account_ratio` = `num_actv_rev_tl / total_acc` | 0.7225 | 0.4496 | ROC-AUC -0.0002 (vs step 3), +0.0001 (vs baseline); F1 +0.0001/+0.0003 |
| 5 | + `purpose_grouped` (14 categories -> 5: debt_related, major_purchase, business, discretionary, other) | 0.7228 | 0.4503 | ROC-AUC +0.0003 (vs step 4), +0.0004 (vs baseline); F1 +0.0008/+0.0011 |

### Round 2 conclusion

All 5 deferred features landed within +/-0.0004 ROC-AUC of baseline - noise-level, none individually meaningful. Combined effect (step 5 vs step 0): +0.0004 ROC-AUC, +0.0011 F1. `purpose_grouped` was the (marginally) best performer.

**This confirms rather than contradicts the hyperparameter tuning finding.** Two independent investigations - tuning (converged ~0.72-0.723 with diminishing returns) and now feature engineering (5 reasonable, domain-motivated features adding essentially nothing) - point the same direction: ~0.72 ROC-AUC looks like a genuine ceiling for this feature set + Random Forest combination, not an artifact of insufficient tuning or an incomplete feature list.

**What this likely means:** the existing ~71-76 features (grade, FICO, DTI, utilization, delinquency counts, income) already contain most of the linearly/tree-discoverable signal Lending Club's origination-time data offers for predicting default. Getting meaningfully past 0.72 probably requires either (a) fundamentally different information not in this dataset, or (b) accepting that origination-time-only prediction has an inherent ceiling and that behavioral/mid-loan-life signals (deferred to survival analysis/uplift modeling per the original project plan) are where the bigger gains live for the broader collections decision engine.

**Decision:** stop iterating on the initial classifier's feature set and hyperparameters for now. The 5 Round 2 features are kept (none hurt, `purpose_grouped` and `sub_grade_numeric` gave the mildest positive signal) but this is diminishing-returns territory. Move on to `04_survival_analysis.ipynb` as originally planned - that's where the hardship/settlement features, time-to-event modeling, and mid-loan-life behavior (deferred throughout preprocessing) become available and relevant.
