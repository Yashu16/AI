## Lending club data dictionary

### Bucket 1 (Target variable)
**loan_status**:  
Fully Paid            - 0     
Current               - IGNORE                      
Charged Off           - 1                                 
Late (31-120 days)    - 1                       
In Grace Period       - 0                         
Late (16-30 days)     - 0                             
Does not meet the credit policy. Status:Fully Paid - but old policy - 0
Does not meet the credit policy. Status:Charged Off - but old policy - 1
Default                - 1

1 - Default; 0 - Did not default 

**Reasoning for my decisions**
- From a business perspective, borrowers in grace period and late period(only 16-30) should not be considered as defaulted. For the initial classifier analysis, we will mark them as not defaulted, but their value is delivered for survival analysis. Borrowers who are late than 30 days are heading towards default, but we don't know the outcome for sure. So, we will right-censor it. For initial classifier analysis, we will exclude rows with that label, and use that in our survival analysis. 

### Bucket 2 (Time columns) (For survival analysis)
**issue_d** - Month/year the loan was funded 
**last_pymnt_d** - Date of last payment made —  "last contact" date
**next_pymnt_d** - Scheduled next payment date
**earliest_cr_line** - When borrower's first credit line was opened (credit age)
**last_credit_pull_d** - Last time LC pulled credit

### Bucket 3 (Payment behavior signals)
**installment** - The monthly amount the borrower should be paying
**last_pymnt_amnt** - The amount they actually paid last
**total_pymnt** - Total amount paid so far across all payments
**total_rec_prncp** - Of total paid, how much went to principal
**total_rec_int** - Of total paid, how much went to interest
**total_rec_late_fee** - Total late fees they've paid — a strong signal
**out_prncp** - Remaining principal still owed
**recoveries** - Amount recovered after charge-off — post-default signal
**delinq_2yrs** - Number of times 30+ days past due in last 2 years
**mths_since_last_delinq** - How long ago was the last missed payment 

**NOTE:** There are some columns that can only be filled after we have known the tenant has defaulted, so if not handled carefully, it will lead to data leakage. `I will write it down in near-future`.

### Bucket 4 (Borrower Financial profile) (Risk context)
**loan_amnt** - Amount borrowed
**int_rate** - Interest rate assigned — higher = riskier borrower
**grade** - LC's credit grade: A (safest) through G (riskiest)
**sub_grade** - More granular version: A1, A2... G5
**dti** - Debt-to-income ratio — how stretched is this person financially
**annual_inc** - Self-reported income
**verification_status** - Was income verified? (`Verified`, `Source Verified`, `Not Verified`)
**emp_length** - Employment length 0–10 years
**home_ownership** - `RENT`, `OWN`, `MORTGAGE`, `OTHER`
**open_acc** - Number of open credit lines — fragmentation of debt
**revol_util** - How much of revolving credit is being used (%)
**revol_bal** - Total revolving balance
**pub_rec** - Number of derogatory public records (judgments, liens)
**pub_rec_bankruptcies** - Bankruptcy history
**mort_acc** - Number of mortgage accounts
**inq_last_6mths** - Credit inquiries in last 6 months — signals financial stress

### Bucket 5 (Hardship & Collections signals)
**hardship_flag** - Was a hardship plan ever activated?
**hardship_type** - What kind of hardship plan
**hardship_reason** - Why hardship was claimed
**hardship_start_date / hardship_end_date** - Duration of hardship plan
**debt_settlement_flag** - Did they settle for less than owed?
**debt_settlement_flag_date** - When settlement was agreed
**settlement_date** - When settlement was completed
