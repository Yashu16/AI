# 01_eda.ipynb - Decision Log

## EDA planning
How did I choose the features?
- Since there are about 151 columns, it's not possible to go through every one of them manually without feeling overwhelmed. So, I asked Claude to web search and give me few important columns to start with. It gave me columns separated by different buckets based on the information that I previously gave it - that I will perform survival analysis and uplift modeling. I have written my initial chosen columns in data_dictionary.md.

### When to use SMOTE / resampling
A rough mental model:

| Imbalance Ratio | Treatment |
|---|---|
| 1:2 to 1:5 | `class_weight='balanced'` is enough. |
| 1:5 to 1:20 | Start considering mild resampling — undersample majority, or try SMOTE |
| 1:20 to 1:100 | SMOTE or other oversampling becomes necessary |
| 1:100+ (fraud, rare disease) | Aggressive techniques — SMOTE + undersampling combined, anomaly detection framing, or treat it as a different problem entirely |

- After making a few decisions in data_dictionary.md related to the target variable, I got a 1:3.75 class split ratio. Since this is a good ratio, I will simply use `balanced` as class_weight when doing sklearn.
- Metrics: choosing a balance between recall and precision is important, so ROC-AUC is our best metric here especially because tuning the decision threshold becomes important later on for collections decisions. F1-score is a secondary metric.

## Preprocessing - Step 1: Column drops

Went through every column to make sure nothing was missed.

### Reason 1: Too many nulls (>=70% missing)
Dropped (hardship/settlement/joint-applicant fields, deferred to survival/uplift analysis later):
`member_id, orig_projected_additional_accrued_interest, hardship_reason, hardship_payoff_balance_amount, hardship_last_payment_amount, payment_plan_start_date, hardship_type, hardship_status, hardship_start_date, deferral_term, hardship_amount, hardship_dpd, hardship_loan_status, hardship_length, hardship_end_date, settlement_status, debt_settlement_flag_date, settlement_term, settlement_percentage, settlement_date, settlement_amount, sec_app_mths_since_last_major_derog, sec_app_revol_util, revol_bal_joint, sec_app_inq_last_6mths, sec_app_num_rev_accts, sec_app_open_acc, sec_app_earliest_cr_line, sec_app_fico_range_high, sec_app_mort_acc, sec_app_open_act_il, sec_app_fico_range_low, sec_app_collections_12_mths_ex_med, sec_app_chargeoff_within_12_mths, verification_status_joint, dti_joint, annual_inc_joint, desc, mths_since_last_record, mths_since_recent_bc_dlq, mths_since_last_major_derog`

### 40-70% missing band
`mths_since_recent_revol_delinq(67.25), next_pymnt_d(59.51), mths_since_last_delinq(51.25), il_util(47.28), mths_since_rcnt_il(40.25)`

Went through each individually - only `mths_since_last_delinq` (Bucket 3) is worth keeping (flag + sentinel, see Step 3 imputation below). The other four don't carry unique signal beyond what's already kept - dropped.

**Note on `next_pymnt_d`:** dropped for a different reason than plain high-missingness. It's structurally near-always null for closed loans (no "next" payment once a loan has ended), so within our classifier subset (already-concluded loans only) its missingness isn't informative - and the rare non-null values would leak information about loan status. Not a bucket-membership decision, a structural one.

**Correction:** after building `clf_df` and re-checking remaining nulls post-Step 3, found `all_util` (58.8% missing in `clf_df`) had been mistakenly left out of this drop list and almost got swept into the negligible-missing "row drop" step instead - which would have dropped ~59% of rows. Added `all_util` to this band's drop list. **Lesson: always double check a column's actual missing count before bucketing it as "negligible."**

### 38.3% missing band
`open_acc_6m, inq_last_12m, total_cu_tl, open_il_24m, max_bal_bc, open_act_il, open_il_12m, open_rv_24m, total_bal_il, inq_fi, open_rv_12m`

Credit bureau trade line/inquiry snapshot fields (recent account openings, inquiries, balances). Missingness here is a data-collection-era artifact (LC only started pulling this data partway through its history), not borrower behavior - so unlike the 40-70% band, a "was missing" flag would just encode loan age, not signal. Also largely duplicate signal already in Bucket 4 (`open_acc`, `inq_last_6mths`, `revol_bal`). Dropped for the initial simple model; revisit later if it underperforms.

### 13.07% missing: `mths_since_recent_inq`
Months since most recent credit inquiry - similar in spirit to `inq_last_6mths` (Bucket 4). NaN likely means no inquiries on record, same pattern as `mths_since_last_delinq`. Kept - flag + sentinel (see Step 3).

### ~6.5-7.4% missing: `emp_title(7.39), num_tl_120dpd_2m(6.80), emp_length(6.50)`
- `emp_title`: free-text self-reported job title, extremely messy/high-cardinality. Dropped for now, not usable without heavy NLP grouping.
- `num_tl_120dpd_2m`: number of trade lines 120+ days past due in last 2 months - strong delinquency signal, similar to `delinq_2yrs` (Bucket 3). Kept, sentinel.
- `emp_length`: employment length in years (0-10). Kept, NaN treated as an "unknown" category rather than dropping rows.

### 6.15% missing: `mo_sin_old_il_acct`
Months since oldest installment account opened - measure of installment credit history length. Kept, NaN filled with 0/sentinel.

### ~2.2%-3.4% missing (credit bureau summary stats, ~35 columns)
`bc_util, percent_bc_gt_75, bc_open_to_buy, pct_tl_nvr_dlq, total_rev_hi_lim, tot_hi_cred_lim, total_il_high_credit_limit, total_bal_ex_mort, total_bc_limit, avg_cur_bal, tot_cur_bal, tot_coll_amt, mths_since_recent_bc, mo_sin_rcnt_rev_tl_op, mo_sin_old_rev_tl_op, mo_sin_rcnt_tl, num_rev_accts, num_op_rev_tl, num_il_tl, num_actv_rev_tl, num_actv_bc_tl, num_bc_tl, num_bc_sats, num_sats, num_rev_tl_bal_gt_0, acc_open_past_24mths, mort_acc, num_tl_op_past_12m, num_tl_30dpd, num_tl_90g_dpd_24m, num_accts_ever_120_pd`

All standard credit-risk summary stats (utilization/limits, account age/recency, account counts, delinquency counts) - overlaps conceptually with Bucket 4. Missingness (2-3%) plausibly means "borrower has zero relevant trade lines" rather than a data problem. Kept all, filled NaN with 0. Note: lots of near-duplicate/overlapping features here (`num_sats` vs `num_bc_sats`, `tot_cur_bal` vs `avg_cur_bal`) - worth a multicollinearity check later.

### Reason 2: Not useful regardless of missing %
`title, id, url, policy_code, zip_code, pymnt_plan, initial_list_status, disbursement_method`

- `title`: free-text loan title, redundant with `purpose` (categorical).
- `id`: unique identifier, no predictive value.
- `url`: just a link to the loan page, no information.
- `zip_code`: truncated to 3 digits, low-resolution; `addr_state` already gives state-level geography.
- `policy_code`: LC internal flag, almost always constant (1.0), no variance.
- `pymnt_plan`: almost always "n", near-zero variance.
- `initial_list_status`: whole vs fractional listing, internal LC platform detail, not borrower behavior.
- `disbursement_method`: Cash vs DirectPay, mostly Cash, low variance/minor signal.

Dropped all of these regardless of their negligible missing %.

Date fields (`last_pymnt_d`, `last_credit_pull_d`, `earliest_cr_line`, `issue_d`) are fine and useful (Bucket 2) - just need datetime parsing, not dropping.

## Step 2: Target + classifier subset
Applied Bucket 1 label mapping (see data_dictionary.md) to create `target` (0/1/CENSOR). Built `clf_df = df[df['target'] != 'CENSOR']` - classifier-eligible subset excludes `Current` loans.

Result: Total rows 2,260,701. Full grouping: 0 (48.28%), CENSOR (38.85%), 1 (12.86%). Classifier subset: 1,382,384 rows (61.15% of full data), Class 0: 78.96%, Class 1: 21.04%, imbalance ratio 1:3.75.

## Step 3: Missing value imputation (on `clf_df`)

- **"Absence is informative" columns** (`mths_since_last_delinq`, `mths_since_recent_inq`, `num_tl_120dpd_2m`, `mo_sin_old_il_acct`): sentinel fill (999) + explicit `_missing_flag` column.
- **`emp_length`**: filled with `'unknown'` category.
- **Credit-bureau summary-stat cluster** (~35 cols): zero fill.
- **`emp_title`**: dropped (see above).
- **Remaining negligible-missing columns** (<=0.18% each, at most ~2,460 rows out of 1.38M, largely overlapping bad/incomplete CSV rows): dropped the rows rather than imputing.

Result: shape (1,377,201, 90), zero remaining nulls.

## Step 4: Leakage audit

Went through every remaining feature-candidate column (84 total) to decide which are known only after a loan concludes (leakage) vs safe to use.

**Leaky - excluded from classifier features:**
- `recoveries`, `collection_recovery_fee`: only populated once a loan has already defaulted and gone to collections. Practically encodes the target.
- `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`, `out_prncp`, `out_prncp_inv`: cumulative totals over the entire loan life - directly reflect how the loan ended.
- `last_pymnt_amnt`, `last_pymnt_d`: summarize the final payment, fundamentally different in character for a defaulted vs completed loan.
- `last_fico_range_low`, `last_fico_range_high`, `last_credit_pull_d`: FICO/credit pull *after* the loan concluded.
- `hardship_flag`: whether a hardship plan was ever activated - a mid-loan event, often granted when a borrower is already struggling.
- `debt_settlement_flag`: settling for less than owed is itself close to a default-adjacent event.
- `funded_amnt_inv`: nearly always equals `loan_amnt`/`funded_amnt`; low risk but redundant - use `loan_amnt` instead.

**Correction:** the first pass was driven only by the "remaining nulls after Step 3" list, which misses zero-null columns (`hardship_flag`, `debt_settlement_flag`, `funded_amnt_inv` were caught on a second, full pass over all 84 columns). **Lesson: missingness-driven review and leakage review are different passes - a column can be 0% missing and still leak the outcome.**

**Borderline (bureau-pull-time snapshots, believed safe but not yet empirically verified):**
`chargeoff_within_12_mths, collections_12_mths_ex_med, acc_now_delinq, delinq_amnt, total_rev_hi_lim, tot_hi_cred_lim, tot_cur_bal, total_bal_ex_mort, total_bc_limit, total_il_high_credit_limit, avg_cur_bal, bc_open_to_buy, bc_util, percent_bc_gt_75, pct_tl_nvr_dlq, num_accts_ever_120_pd, num_actv_bc_tl, num_actv_rev_tl, num_bc_sats, num_bc_tl, num_il_tl, num_op_rev_tl, num_rev_accts, num_rev_tl_bal_gt_0, num_sats, num_tl_30dpd, num_tl_90g_dpd_24m, num_tl_op_past_12m, mo_sin_old_rev_tl_op, mo_sin_rcnt_rev_tl_op, mo_sin_rcnt_tl, mort_acc, mths_since_recent_bc, mths_since_recent_inq, acc_open_past_24mths`

**Empirical check:** compared mean/median by target class. All 34 borderline columns showed target_1/target_0 ratios roughly between 0.7 and 1.4 - mild, domain-sensible differences, consistent with legitimate credit-risk signal rather than leakage. Cleared all 34 as safe classifier features.

**Positive control:** ran the same check on known-leaky columns (`total_pymnt`, `total_rec_prncp`, `out_prncp`, `recoveries`, `collection_recovery_fee`, `last_pymnt_amnt`) to validate the method. Ratios were far outside the safe 0.7-1.4 band: `out_prncp` 6.57x, `last_pymnt_amnt` 0.073x, `total_rec_prncp` 0.32x, and `recoveries`/`collection_recovery_fee` had target_0 pinned at exactly 0 (ratio undefined). Confirms the check reliably distinguishes leaky columns from mild-signal-but-safe ones. **Reference calibration for future projects: ratios beyond roughly 0.5-2x, or one class pinned at a constant, are the red flag threshold, not mild 0.7-1.4 variation.**

## Step 5: Train/val/test split

Decision: random stratified split (on `target`), 70/15/15. Two-step split via sklearn's `train_test_split` (split off test first, then split remaining into train/val), `stratify=target`.

**Future work** (revisit once the simple model is working): time-based split by `issue_d` (train on earlier-issued loans, test on later-issued ones) - more realistic for eventual deployment since loans span 2007-2018 with different economic conditions (e.g. 2008 crash). Also flagged for later: revisit hardship/settlement features and any other leakage-sensitive decisions made for simplicity now, once survival analysis/uplift/deployment stages begin.

**Result:** Train (964040, 71), Val (206580, 71), Test (206581, 71). Class balance held at 79.10%/20.90% across all three splits - stratification worked as intended.

Saved to `data/processed/train.parquet`, `val.parquet`, `test.parquet` for handoff to `02_feat_engineer.ipynb`.

**Tooling note:** hit a pyarrow/pandas `ArrowKeyError` (stale extension-type registration) on first save attempt after installing `pyarrow` mid-session - fixed by a full kernel restart.
