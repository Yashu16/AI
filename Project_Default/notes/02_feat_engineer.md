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
