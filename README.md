# Customer Churn Prediction — Telco Dataset

Predicts which telecom customers are likely to churn using real customer data, and identifies the key business drivers behind churn.

## What this project demonstrates
- Real-world data cleaning (handling a genuine missing-data edge case: blank `TotalCharges` values tied to brand-new customers with zero tenure)
- Exploratory data analysis on churn drivers (contract type, tenure, monthly charges, internet service)
- Feature engineering: binary encoding + one-hot encoding of categorical variables
- Classification modeling: Logistic Regression (baseline, scaled features) vs. Random Forest
- Model evaluation: accuracy, precision, recall, F1, confusion matrix, ROC-AUC
- Feature importance analysis to translate model output into a business recommendation

## Data source
**IBM Telco Customer Churn dataset** — a well-known, publicly available, real dataset of 7,043 telecom customers. Source: [IBM's public GitHub repository](https://github.com/IBM/telco-customer-churn-on-icp4d). This is real, not synthetic, data.

## Results
- Churn rate in the dataset: **26.5%**
- Random Forest model: **~80% accuracy**, meaningful ROC-AUC separation between churners and non-churners
- Top churn drivers: low tenure, month-to-month contracts, fiber optic internet service, electronic check payment method

## Repo structure
```
churn-project/
├── data/
│   ├── telco_churn_raw.csv
│   └── telco_churn_clean.csv
├── charts/                        # exported PNG charts
├── churn_analysis.py              # full pipeline script
├── notebook/
│   └── churn_prediction.ipynb     # full notebook with narrative + outputs
└── README.md
```

## How to reproduce
```bash
pip install pandas matplotlib seaborn scikit-learn
python3 churn_analysis.py
# or open notebook/churn_prediction.ipynb directly
```

## Tech stack
Python, pandas, matplotlib, seaborn, scikit-learn (Logistic Regression, Random Forest)
