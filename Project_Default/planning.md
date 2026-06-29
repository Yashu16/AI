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