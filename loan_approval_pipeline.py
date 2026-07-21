"""
Loan Approval Prediction with Machine Learning
================================================
End-to-end pipeline: EDA -> preprocessing -> model training ->
evaluation -> model comparison -> feature importance -> save best model.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
import joblib

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("loan_prediction.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
print("Dataset shape:", df.shape)
print(df.head())

# ---------------------------------------------------------------
# 2. HANDLE MISSING VALUES
# ---------------------------------------------------------------
cat_cols_na = ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]
for col in cat_cols_na:
    df[col] = df[col].fillna(df[col].mode()[0])

df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])

print("\nMissing values after imputation:\n", df.isnull().sum().sum())

# ---------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["LoanAmount_log"] = np.log1p(df["LoanAmount"])
df["TotalIncome_log"] = np.log1p(df["TotalIncome"])
df["EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"]
df["Balance_Income"] = df["TotalIncome"] - (df["EMI"] * 1000)
df["Dependents"] = df["Dependents"].replace("3+", "3").astype(int)

# ---------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (saved as PNGs)
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

sns.countplot(x="Loan_Status", data=df, ax=axes[0, 0], palette="viridis")
axes[0, 0].set_title("Loan Status Distribution")

sns.countplot(x="Credit_History", hue="Loan_Status", data=df, ax=axes[0, 1], palette="viridis")
axes[0, 1].set_title("Credit History vs Loan Status")

sns.countplot(x="Education", hue="Loan_Status", data=df, ax=axes[0, 2], palette="viridis")
axes[0, 2].set_title("Education vs Loan Status")

sns.countplot(x="Property_Area", hue="Loan_Status", data=df, ax=axes[1, 0], palette="viridis")
axes[1, 0].set_title("Property Area vs Loan Status")

sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=df, ax=axes[1, 1], palette="viridis")
axes[1, 1].set_title("Applicant Income vs Loan Status")
axes[1, 1].set_ylim(0, 20000)

sns.boxplot(x="Loan_Status", y="LoanAmount", data=df, ax=axes[1, 2], palette="viridis")
axes[1, 2].set_title("Loan Amount vs Loan Status")

plt.tight_layout()
plt.savefig("eda_overview.png", bbox_inches="tight")
plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 7))
num_df = df.select_dtypes(include=[np.number])
sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", bbox_inches="tight")
plt.close()

print("\nSaved EDA plots: eda_overview.png, correlation_heatmap.png")

# ---------------------------------------------------------------
# 5. ENCODE CATEGORICAL VARIABLES
# ---------------------------------------------------------------
df_model = df.drop(columns=["Loan_ID"])

label_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le

target_le = LabelEncoder()
df_model["Loan_Status"] = target_le.fit_transform(df_model["Loan_Status"])  # N=0, Y=1

# ---------------------------------------------------------------
# 6. TRAIN / TEST SPLIT
# ---------------------------------------------------------------
X = df_model.drop(columns=["Loan_Status"])
y = df_model["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ---------------------------------------------------------------
# 7. TRAIN MULTIPLE MODELS
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=9),
}

results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
fitted_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    fitted_models[name] = model
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring="accuracy")

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
        "CV_Mean_Acc": cv_scores.mean(),
        "CV_Std": cv_scores.std(),
    })

results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
print("\n=== Model Comparison ===")
print(results_df.round(4).to_string(index=False))
results_df.to_csv("model_comparison.csv", index=False)

best_model_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_model_name]
print(f"\nBest model: {best_model_name}")

# ---------------------------------------------------------------
# 8. DETAILED EVALUATION OF BEST MODEL
# ---------------------------------------------------------------
y_pred_best = best_model.predict(X_test_scaled)
y_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report (Best Model):")
print(classification_report(y_test, y_pred_best, target_names=["Rejected (N)", "Approved (Y)"]))

cm = confusion_matrix(y_test, y_pred_best)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"], ax=axes[0])
axes[0].set_title(f"Confusion Matrix - {best_model_name}")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

fpr, tpr, _ = roc_curve(y_test, y_proba_best)
axes[1].plot(fpr, tpr, label=f"{best_model_name} (AUC={roc_auc_score(y_test, y_proba_best):.3f})", color="darkorange")
axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend()

plt.tight_layout()
plt.savefig("best_model_evaluation.png", bbox_inches="tight")
plt.close()

# Model comparison bar chart
plt.figure(figsize=(10, 6))
plot_df = results_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"])
sns.barplot(data=plot_df, x="Model", y="value", hue="variable", palette="viridis")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 1)
plt.title("Model Comparison Across Metrics")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("model_comparison_chart.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# 9. FEATURE IMPORTANCE (tree-based, fallback to Random Forest)
# ---------------------------------------------------------------
importance_source = fitted_models.get("Random Forest")
importances = pd.Series(importance_source.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(9, 6))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", bbox_inches="tight")
plt.close()

print("\nTop features:\n", importances.head(6))

# ---------------------------------------------------------------
# 10. SAVE ARTIFACTS
# ---------------------------------------------------------------
joblib.dump(best_model, "best_loan_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(encoders, "label_encoders.joblib")
joblib.dump(target_le, "target_encoder.joblib")
joblib.dump(list(X.columns), "feature_columns.joblib")

print("\nSaved model artifacts: best_loan_model.joblib, scaler.joblib, label_encoders.joblib, target_encoder.joblib, feature_columns.joblib")
print("\nDONE.")
