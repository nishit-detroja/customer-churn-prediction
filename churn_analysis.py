"""
churn_analysis.py
-------------------
Customer Churn Prediction - Telco dataset (IBM public dataset, real data).

Pipeline: load -> clean -> EDA -> feature engineering -> classification models
(Logistic Regression + Random Forest) -> evaluation -> feature importance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_curve, auc)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

RAW = "/home/claude/churn-project/data/telco_churn_raw.csv"
CLEAN = "/home/claude/churn-project/data/telco_churn_clean.csv"
CHART_DIR = "/home/claude/churn-project/charts"

# ---------- 1. Load & clean ----------
df = pd.read_csv(RAW)
print("Raw shape:", df.shape)

# TotalCharges has 11 blank strings for customers with tenure=0 (brand new
# customers who haven't been billed yet) -- a real, documented edge case.
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Blank TotalCharges rows (tenure=0 customers):", df["TotalCharges"].isna().sum())

# These 11 rows have tenure=0, so TotalCharges should logically be 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Drop customerID (identifier, not a feature)
df = df.drop(columns=["customerID"])

df.to_csv(CLEAN, index=False)
print("Cleaned shape:", df.shape)
print("Churn rate:", round(df["Churn"].eq("Yes").mean() * 100, 1), "%")

# ---------- 2. EDA ----------

# 2.1 Overall churn distribution
plt.figure(figsize=(6, 5))
churn_counts = df["Churn"].value_counts()
colors = ["#2E86AB", "#E63946"]
plt.pie(churn_counts.values, labels=churn_counts.index, autopct="%1.1f%%",
        colors=colors, startangle=90)
plt.title("Overall Customer Churn Rate")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/churn_overall.png")
plt.close()

# 2.2 Churn by contract type
plt.figure(figsize=(8, 5))
contract_churn = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
contract_churn.plot(kind="bar", stacked=True, color=["#2E86AB", "#E63946"], ax=plt.gca())
plt.title("Churn Rate by Contract Type")
plt.ylabel("Percentage")
plt.xlabel("Contract Type")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/churn_by_contract.png")
plt.close()

# 2.3 Tenure distribution by churn
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack",
             palette=["#2E86AB", "#E63946"], bins=30)
plt.title("Customer Tenure Distribution by Churn Status")
plt.xlabel("Tenure (months)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/tenure_by_churn.png")
plt.close()

# 2.4 Monthly charges by churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn",
            palette=["#2E86AB", "#E63946"], legend=False)
plt.title("Monthly Charges Distribution by Churn Status")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/charges_by_churn.png")
plt.close()

# 2.5 Churn by internet service type
plt.figure(figsize=(8, 5))
internet_churn = pd.crosstab(df["InternetService"], df["Churn"], normalize="index") * 100
internet_churn.plot(kind="bar", stacked=True, color=["#2E86AB", "#E63946"], ax=plt.gca())
plt.title("Churn Rate by Internet Service Type")
plt.ylabel("Percentage")
plt.xlabel("Internet Service")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/churn_by_internet.png")
plt.close()

# ---------- 3. Feature engineering ----------
df_model = df.copy()
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
for col in binary_cols:
    df_model[col] = df_model[col].map({"Yes": 1, "No": 0})

df_model["gender"] = df_model["gender"].map({"Male": 1, "Female": 0})

categorical_cols = ["MultipleLines", "InternetService", "OnlineSecurity",
                     "OnlineBackup", "DeviceProtection", "TechSupport",
                     "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"]

df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

X = df_model.drop(columns=["Churn"])
y = df_model["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------- 4. Models ----------
# Logistic Regression benefits from scaled features (numeric columns like
# tenure/charges are on very different scales from the 0/1 dummy columns)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)

rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

def report(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
    print(f"F1-score:  {f1_score(y_true, y_pred):.3f}")

report("Logistic Regression", y_test, y_pred_lr)
report("Random Forest", y_test, y_pred_rf)

# ---------- 5. Confusion matrix (Random Forest, the better-suited model here) ----------
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix - Random Forest")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/confusion_matrix.png")
plt.close()

# ---------- 6. ROC curve ----------
y_prob_rf = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="#E63946", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest Churn Model")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/roc_curve.png")
plt.close()

# ---------- 7. Feature importance ----------
importances = pd.Series(rf.feature_importances_, index=X.columns, name="importance").sort_values(ascending=False).head(15)

plt.figure(figsize=(9, 6))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
            palette="rocket", legend=False)
plt.title("Top 15 Features Driving Churn Prediction (Random Forest)")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/feature_importance.png")
plt.close()

print("\nTop 10 churn-driving features:")
print(importances.head(10))
print("\nAll charts saved to", CHART_DIR)
