"""
Boston House Price Prediction
==============================
End-to-end regression pipeline: data preprocessing, model selection,
training, and evaluation.

Dataset columns:
crim    - per capita crime rate by town
zn      - proportion of residential land zoned for large lots
indus   - proportion of non-retail business acres per town
chas    - Charles River dummy variable (1 if bounds river, else 0)
nox     - nitric oxide concentration
rm      - average number of rooms per dwelling
age     - proportion of owner-occupied units built before 1940
dis     - weighted distance to employment centers
rad     - index of accessibility to radial highways
tax     - property tax rate
ptratio - pupil-teacher ratio by town
b       - 1000(Bk - 0.63)^2, Bk = proportion of Black residents
lstat   - % lower status of the population
medv    - median value of owner-occupied homes in $1000s (TARGET)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
sns.set_style("whitegrid")

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("BostonHousing.csv")
df.columns = [c.strip().lower() for c in df.columns]
df["chas"] = df["chas"].astype(int)

print("Shape:", df.shape)
print(df.head())
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDescriptive statistics:\n", df.describe().T)

# ---------------------------------------------------------------
# 2. DATA PREPROCESSING
# ---------------------------------------------------------------
# 2a. Handle missing values (if any) via median imputation
if df.isnull().sum().sum() > 0:
    df = df.fillna(df.median(numeric_only=True))

# 2b. Outlier check on target (cap known censoring artifact at medv==50)
print("\nRows with medv == 50 (censored max value):", (df["medv"] == 50).sum())
df = df[df["medv"] < 50].reset_index(drop=True)  # remove censored top-coded values

# 2c. Correlation heatmap (saved to file)
plt.figure(figsize=(11, 9))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("/home/claude/correlation_heatmap.png", dpi=150)
plt.close()

print("\nCorrelation with target (medv), sorted:\n",
      corr["medv"].sort_values(ascending=False))

# 2d. Feature / target split
X = df.drop(columns=["medv"])
y = df["medv"]

# 2e. Train/test split (done BEFORE scaling to avoid data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# 2f. Feature scaling (fit only on training data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# ---------------------------------------------------------------
# 3. MODEL SELECTION & TRAINING
# ---------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Lasso Regression": Lasso(alpha=0.01, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(
        n_estimators=150, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
    ),
    "SVR (RBF)": SVR(kernel="rbf", C=10, epsilon=0.5),
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, cv=5, scoring="r2"
    )

    results.append({
        "Model": name,
        "Test RMSE": round(rmse, 3),
        "Test MAE": round(mae, 3),
        "Test R2": round(r2, 3),
        "CV R2 (mean)": round(cv_scores.mean(), 3),
        "CV R2 (std)": round(cv_scores.std(), 3),
    })

results_df = pd.DataFrame(results).sort_values("Test R2", ascending=False)
print("\n===== Model Comparison =====")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------
# 4. HYPERPARAMETER TUNING ON BEST MODEL FAMILY (Random Forest)
# ---------------------------------------------------------------
param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [None, 10, 16],
    "min_samples_split": [2, 4],
    "min_samples_leaf": [1, 2],
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1,
)
grid.fit(X_train_scaled, y_train)

print("\nBest Random Forest params:", grid.best_params_)
print("Best CV R2:", round(grid.best_score_, 3))

best_model = grid.best_estimator_
final_preds = best_model.predict(X_test_scaled)

final_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
final_mae = mean_absolute_error(y_test, final_preds)
final_r2 = r2_score(y_test, final_preds)

print(f"\n===== Final Tuned Model (Random Forest) =====")
print(f"Test RMSE: {final_rmse:.3f}")
print(f"Test MAE : {final_mae:.3f}")
print(f"Test R2  : {final_r2:.3f}")

# ---------------------------------------------------------------
# 5. FEATURE IMPORTANCE
# ---------------------------------------------------------------
importances = pd.Series(best_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(9, 6))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
            palette="viridis", legend=False)
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("/home/claude/feature_importance.png", dpi=150)
plt.close()

print("\nFeature importances:\n", importances)

# ---------------------------------------------------------------
# 6. ACTUAL VS PREDICTED PLOT
# ---------------------------------------------------------------
plt.figure(figsize=(7, 7))
plt.scatter(y_test, final_preds, alpha=0.6, edgecolor="k")
lims = [min(y_test.min(), final_preds.min()), max(y_test.max(), final_preds.max())]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual MEDV ($1000s)")
plt.ylabel("Predicted MEDV ($1000s)")
plt.title(f"Actual vs Predicted (Tuned Random Forest, R2={final_r2:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig("/home/claude/actual_vs_predicted.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 7. SAVE RESULTS TABLE
# ---------------------------------------------------------------
results_df.to_csv("/home/claude/model_comparison.csv", index=False)

print("\nAll artifacts saved: correlation_heatmap.png, feature_importance.png,")
print("actual_vs_predicted.png, model_comparison.csv")
