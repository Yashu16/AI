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

Next, we shall handle missing values for existing columns. And then we can feature engineer from them. 

**Feature engineering**
Time to engineer new features from existing ones. 