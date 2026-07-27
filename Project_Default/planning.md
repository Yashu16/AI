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

These following columns have too many missing values (>=74%) and are not useful for our analysis, so we will drop them:
member_id, orig_projected_additional_accrued_interest, hardship_reason, hardship_payoff_balance_amount, hardship_last_payment_amount, payment_plan_start_date, hardship_type, hardship_status, hardship_start_date, deferral_term, hardship_amount, hardship_dpd, hardship_loan_status, hardship_length, hardship_end_date, settlement_status, debt_settlement_flag_date, settlement_term, settlement_percentage, settlement_date, settlement_amount, sec_app_mths_since_last_major_derog, sec_app_revol_util, revol_bal_joint, sec_app_inq_last_6mths, sec_app_num_rev_accts, sec_app_open_acc, sec_app_earliest_cr_line, sec_app_fico_range_high, sec_app_mort_acc, sec_app_open_act_il, sec_app_fico_range_low, sec_app_collections_12_mths_ex_med, sec_app_chargeoff_within_12_mths, verification_status_joint, dti_joint, annual_inc_joint, desc, mths_since_last_record, mths_since_recent_bc_dlq, mths_since_last_major_derog



Next, we shall handle missing values for existing columns. And then we can feature engineer from them. 

**Feature engineering**
Time to engineer new features from existing ones. 