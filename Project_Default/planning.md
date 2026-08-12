Building a collection decision engine. 

storage-intelligence/
│
├── data/
│   ├── raw/                  # Original downloaded data, never touch it
│   └── processed/            # Your cleaned, feature-engineered data
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_classifier.ipynb
│   ├── 04_survival_analysis.ipynb
│   ├── 05_bayesian_engine.ipynb
│   ├── 06_uplift_model.ipynb
│   └── 07_decision_layer.ipynb
│
├── src/
│   ├── features.py           # Reusable feature engineering functions
│   ├── models.py             # Model training/loading logic
│   └── decision.py           # Decision layer logic
│
├── api/
│   └── main.py               # FastAPI endpoint
│
├── requirements.txt
├── README.md
└── .gitignore

**EDA planning**
How did I choose the features? 
- Since there are about 151 columns, it's not possible to go through every one of them manually without feeling overwhelmed. So, I asked Claude to web search and give me few important columns to start with. It gave me columns separated by different buckets based on the information that I previously gave it - that I will perform survival analysis and uplift modeling. I have written my initial chosen columns in data dictionary md file. 

When to Use SMOTE / Resampling
A rough mental model:

| Imbalance Ratio | Treatment |
|---|---|
| 1:2 to 1:5 | `class_weight='balanced'` is enough. |
| 1:5 to 1:20 | Start considering mild resampling — undersample majority, or try SMOTE |
| 1:20 to 1:100 | SMOTE or other oversampling becomes necessary |
| 1:100+ (fraud, rare disease) | Aggressive techniques — SMOTE + undersampling combined, anomaly detection framing, or treat it as a different problem entirely |

- After making few decisions in data dictionary related to target variable, I got a 1:3.75 class split ratio. Since this is in good ratio, I will simply use `balanced` as class_weight when doing Sk-learn. 

- Next as for metrics: Choosing a balance between recall and precision is important, so ROC-AUC curve is our best metric here especially because tuning the decision threshold becomes important later on for collections decision. Another metric we can use is F1-score. 

**preprocessing**
What columns to drop - and why? 
I will go through each column to make sure we are not missing out on any. 

**Reason 1**: Too many nulls
These following columns have too many missing values (>=70%) and are not useful for our analysis, so we will drop them:
member_id, orig_projected_additional_accrued_interest, hardship_reason, hardship_payoff_balance_amount, hardship_last_payment_amount, payment_plan_start_date, hardship_type, hardship_status, hardship_start_date, deferral_term, hardship_amount, hardship_dpd, hardship_loan_status, hardship_length, hardship_end_date, settlement_status, debt_settlement_flag_date, settlement_term, settlement_percentage, settlement_date, settlement_amount, sec_app_mths_since_last_major_derog, sec_app_revol_util, revol_bal_joint, sec_app_inq_last_6mths, sec_app_num_rev_accts, sec_app_open_acc, sec_app_earliest_cr_line, sec_app_fico_range_high, sec_app_mort_acc, sec_app_open_act_il, sec_app_fico_range_low, sec_app_collections_12_mths_ex_med, sec_app_chargeoff_within_12_mths, verification_status_joint, dti_joint, annual_inc_joint, desc, mths_since_last_record, mths_since_recent_bc_dlq, mths_since_last_major_derog

For our initial analysis, we will drop these columns(hardhsip and settlement). But in future, we can use them for surival analysis and uplift modeling.

Columns with missing values between 40-70%:
mths_since_recent_revol_delinq(67.25), next_pymnt_d(59.51), mths_since_last_delinq(51.25), il_util(47.28),  mths_since_rcnt_il(40.25)

I can't drop these columns just simply because they have missing values, as they are important for our analysis. Need to check if I can flag them for future important information...

Update: went through each individually - only mths_since_last_delinq (Bucket 3) is worth keeping (flag + sentinel, see Step 3 imputation). The other four (mths_since_recent_revol_delinq, next_pymnt_d, il_util, mths_since_rcnt_il) don't carry unique signal beyond what's already kept - decision: drop.

Note on next_pymnt_d specifically: dropped for a different reason than plain high-missingness. It's structurally near-always null for closed loans (no "next" payment once a loan has ended), so within our classifier subset (already-concluded loans only) its missingness isn't informative - and the rare non-null values would leak information about loan status. Not a bucket-membership decision, a structural one.

Correction: after building clf_df and re-checking remaining nulls post-Step 3, found all_util (58.8% missing in clf_df) had been mistakenly left out of this drop list and almost got swept into the negligible-missing "row drop" step instead - which would have dropped ~59% of rows. Added all_util to this 40-70% band's drop list. Lesson: always double check a column's actual missing count before bucketing it as "negligible."

Columns with missing value at 38.3%:
open_acc_6m, inq_last_12m, total_cu_tl, open_il_24m, max_bal_bc, open_act_il, open_il_12m, open_rv_24m, total_bal_il, inq_fi, open_rv_12m

These are credit bureau trade line/inquiry snapshot fields (recent account openings, inquiries, balances). Missingness here is a data-collection-era artifact (LC only started pulling this data partway through its history), not borrower behavior - so unlike the 40-70% band, a "was missing" flag would just encode loan age, not signal. They also largely duplicate signal already in Bucket 4 (open_acc, inq_last_6mths, revol_bal). Decision: drop this whole band for the initial simple model; revisit later if the simple model underperforms.

Column with missing value at 13.07%:
mths_since_recent_inq

Months since most recent credit inquiry - similar in spirit to inq_last_6mths (Bucket 4), recent inquiry activity is a classic credit-risk signal. NaN likely means no inquiries on record, same pattern as mths_since_last_delinq. Decision: keep. Create a `no_recent_inq` flag for NaNs, then fill NaN with a large sentinel value (no recent inquiry = best-case end of the scale).

Columns with missing values at ~6.5-7.4%:
emp_title(7.39), num_tl_120dpd_2m(6.80), emp_length(6.50)

- emp_title: free-text self-reported job title, extremely messy/high-cardinality. Decision: drop for now, not usable without heavy NLP grouping.
- num_tl_120dpd_2m: number of trade lines 120+ days past due in last 2 months - strong delinquency signal, similar to delinq_2yrs (Bucket 3). Decision: keep, handle NaN with a sentinel.
- emp_length: employment length in years (0-10). Decision: keep, NaN likely means unemployed/unreported - treat as an "unknown" category rather than dropping rows.

Column with missing value at 6.15%:
mo_sin_old_il_acct

Months since oldest installment account opened - measure of installment credit history length. Decision: keep, NaN likely means no installment accounts, fill with 0/sentinel.

Columns with missing values at ~2.2%-3.4% (credit bureau summary stats, ~35 columns):
bc_util, percent_bc_gt_75, bc_open_to_buy, pct_tl_nvr_dlq, total_rev_hi_lim, tot_hi_cred_lim, total_il_high_credit_limit, total_bal_ex_mort, total_bc_limit, avg_cur_bal, tot_cur_bal, tot_coll_amt, mths_since_recent_bc, mo_sin_rcnt_rev_tl_op, mo_sin_old_rev_tl_op, mo_sin_rcnt_tl, num_rev_accts, num_op_rev_tl, num_il_tl, num_actv_rev_tl, num_actv_bc_tl, num_bc_tl, num_bc_sats, num_sats, num_rev_tl_bal_gt_0, acc_open_past_24mths, mort_acc, num_tl_op_past_12m, num_tl_30dpd, num_tl_90g_dpd_24m, num_accts_ever_120_pd

All standard credit-risk summary stats (utilization/limits, account age/recency, account counts, delinquency counts) - overlaps conceptually with Bucket 4. Missingness (2-3%) plausibly means "borrower has zero relevant trade lines" rather than a data problem. Decision: keep all, fill NaN with 0 (no relevant accounts implies 0 for counts/balances/utilization) rather than building 35 separate flags. Note: lots of near-duplicate/overlapping features here (num_sats vs num_bc_sats, tot_cur_bal vs avg_cur_bal) - worth a multicollinearity check later.

**Reason 2**: Not useful regardless of missing %
Went through the remaining near-zero-missing columns (title at 1.03%, everything else at 0.11% and below). Most of this tail is fine (payment/date fields, Bucket 3), but a few columns have little/no predictive value independent of missingness - identifiers, near-constant values, or redundant with a column we're keeping:
title, id, url, policy_code, zip_code, pymnt_plan, initial_list_status, disbursement_method

- title: free-text loan title, redundant with purpose (categorical).
- id: unique identifier, no predictive value.
- url: just a link to the loan page, no information.
- zip_code: truncated to 3 digits, low-resolution; addr_state already gives state-level geography.
- policy_code: LC internal flag, almost always constant (1.0), no variance.
- pymnt_plan: almost always "n", near-zero variance (verify before finalizing).
- initial_list_status: whole vs fractional listing, internal LC platform detail, not borrower behavior.
- disbursement_method: Cash vs DirectPay, mostly Cash, low variance/minor signal.

Decision: drop all of these regardless of their negligible missing %.

Note: date fields (last_pymnt_d, last_credit_pull_d, earliest_cr_line, issue_d) are fine and useful (Bucket 2) - just need datetime parsing, not dropping. Payment-behavior fields (collection_recovery_fee, recoveries, total_rec_late_fee, total_pymnt, total_rec_prncp, total_rec_int, out_prncp, last_pymnt_amnt, last_fico_range_low/high) are legitimate (Bucket 3) but carry leakage risk since several are only fully known after the loan concludes - re-check at feature selection time (see leakage note above, line 39 in data_dictionary.md).

**Reason 3**: Leakage audit - post-outcome payment/credit fields
Before feature engineering, went through every Bucket 3 payment field to decide which are known only after a loan concludes (leakage for a classifier predicting outcome) vs safe to use.

Leaky - exclude from classifier features:
- recoveries, collection_recovery_fee: only populated once a loan has already defaulted and gone to collections. Practically encodes the target.
- total_pymnt, total_pymnt_inv, total_rec_prncp, total_rec_int, total_rec_late_fee, out_prncp, out_prncp_inv: cumulative totals over the entire loan life - directly reflect how the loan ended (e.g. out_prncp = 0 for Fully Paid, nonzero for Charged Off).
- last_pymnt_amnt, last_pymnt_d: summarize the final payment, fundamentally different in character for a defaulted vs completed loan.
- last_fico_range_low, last_fico_range_high, last_credit_pull_d: FICO/credit pull *after* the loan concluded, reflects post-outcome credit state. fico_range_low/high (at-origination FICO, Bucket 4) is the safe substitute.

Safe to keep - known at/near origination or represent ongoing behavior signal without directly encoding the terminal outcome:
- installment, delinq_2yrs, mths_since_last_delinq: origination-time or slow-moving borrower behavior, not outcome summaries.

These leaky columns are not dropped from clf_df outright (may be useful later, e.g. survival/uplift analysis), but are excluded from the feature set used to train the initial classifier.

Correction: the first pass was driven only by the "remaining nulls after Step 3" list, which misses zero-null columns. Did a full pass over all 84 remaining feature-candidate columns (not just the ones with nulls) and found:

Additional leaky columns (zero-null, missed by the nulls-only pass):
- hardship_flag: whether a hardship plan was ever activated - a mid-loan event, often granted when a borrower is already struggling. Leaky, exclude.
- debt_settlement_flag: settling for less than owed is itself close to a default-adjacent event. Leaky, exclude.
- funded_amnt_inv: amount funded by investors - nearly always equals loan_amnt/funded_amnt, any gap could reflect investor sentiment during funding. Low risk but redundant - use loan_amnt instead.

Borderline - bureau-pull-time snapshots, believed safe (should be at/near origination per LC's docs) but not yet empirically verified:
chargeoff_within_12_mths, collections_12_mths_ex_med, acc_now_delinq, delinq_amnt, total_rev_hi_lim, tot_hi_cred_lim, tot_cur_bal, total_bal_ex_mort, total_bc_limit, total_il_high_credit_limit, avg_cur_bal, bc_open_to_buy, bc_util, percent_bc_gt_75, pct_tl_nvr_dlq, num_accts_ever_120_pd, num_actv_bc_tl, num_actv_rev_tl, num_bc_sats, num_bc_tl, num_il_tl, num_op_rev_tl, num_rev_accts, num_rev_tl_bal_gt_0, num_sats, num_tl_30dpd, num_tl_90g_dpd_24m, num_tl_op_past_12m, mo_sin_old_rev_tl_op, mo_sin_rcnt_rev_tl_op, mo_sin_rcnt_tl, mort_acc, mths_since_recent_bc, mths_since_recent_inq, acc_open_past_24mths

Decision: run an empirical check (mean/median by target class) on this borderline group before finalizing the classifier feature set - see 01_eda.ipynb leakage-check cell. A column that looks suspiciously different across target=0 vs target=1 in a way that doesn't make domain sense warrants a closer look before being trusted as safe.

Lesson: missingness-driven review and leakage review are different passes - a column can be 0% missing and still leak the outcome. A full pass must cover every remaining column, not just ones that showed up from an earlier missing-values check.

Next, we shall handle missing values for existing columns. And then we can feature engineer from them. 

**Feature engineering**
Time to engineer new features from existing ones. 