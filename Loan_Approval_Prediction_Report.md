# Loan Approval Prediction with Machine Learning

## Problem Statement
Predict whether a loan application will be **approved (Y)** or **rejected (N)** based on applicant financial history, income, credit rating, employment status, and other attributes — a common fintech decision-support task.

## Dataset
- **614 applications**, 12 predictive features + target (`Loan_Status`)
- Fields: Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
- Class balance: 422 approved (69%) vs 192 rejected (31%) — mild imbalance

## Pipeline
1. **Missing value imputation** — mode for categorical fields, median for LoanAmount, mode for Loan_Amount_Term
2. **Feature engineering**:
   - `TotalIncome` = Applicant + Coapplicant income
   - Log transforms of income/loan amount (reduce skew)
   - `EMI` and `Balance_Income` (affordability signals)
3. **Encoding** — label encoding for categorical variables; StandardScaler for numeric features
4. **Train/test split** — 80/20, stratified by target
5. **Models trained**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN — each evaluated with 5-fold cross-validation plus a held-out test set

## Results (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | **0.846** | 0.837 | 0.965 | 0.896 | **0.837** |
| SVM | 0.854 | 0.832 | 0.988 | 0.903 | 0.820 |
| Logistic Regression | 0.862 | 0.840 | 0.988 | 0.908 | 0.815 |
| Decision Tree | 0.821 | 0.825 | 0.941 | 0.879 | 0.773 |
| KNN | 0.846 | 0.824 | 0.988 | 0.898 | 0.769 |
| Gradient Boosting | 0.813 | 0.830 | 0.918 | 0.872 | 0.757 |

**Best model selected: Random Forest** (highest ROC-AUC, most robust across CV folds).

- Recall on approved loans is very high (0.96) — the model rarely misses a good applicant.
- Precision/recall on **rejected** applicants is weaker (0.88 precision, 0.58 recall) — the model under-flags risky applicants, largely a consequence of the class imbalance (69% approved). In a real deployment, this trade-off matters: missed rejections are costlier for a lender than missed approvals, so threshold tuning or class-weighting would be a natural next step.

## Key Drivers of Approval (Feature Importance)
1. **Credit_History** — by far the strongest predictor (~42% of importance). Applicants with a good credit history are approved far more often.
2. ApplicantIncome, Balance_Income, TotalIncome(_log), EMI — income and affordability measures cluster together as the next tier of importance.
3. Categorical demographic fields (Gender, Married, Education, Property_Area) contribute comparatively little.

## Files Delivered
- `loan_approval_pipeline.py` — full, reproducible pipeline script
- `eda_overview.png`, `correlation_heatmap.png` — exploratory analysis
- `model_comparison_chart.png`, `model_comparison.csv` — all 6 models compared
- `best_model_evaluation.png` — confusion matrix + ROC curve for Random Forest
- `feature_importance.png` — ranked feature importances
- `best_loan_model.joblib`, `scaler.joblib`, `label_encoders.joblib`, `target_encoder.joblib`, `feature_columns.joblib` — the trained model and preprocessing objects, ready to load and use for new predictions

## Using the Saved Model
```python
import joblib
import pandas as pd

model = joblib.load("best_loan_model.joblib")
scaler = joblib.load("scaler.joblib")
encoders = joblib.load("label_encoders.joblib")
target_le = joblib.load("target_encoder.joblib")
feature_cols = joblib.load("feature_columns.joblib")

# new_applicant: a DataFrame with the same engineered columns as training
# encode categoricals with `encoders`, scale with `scaler`, then:
pred = model.predict(scaled_features)
result = target_le.inverse_transform(pred)  # "Y" or "N"
```

## Possible Next Steps
- Address class imbalance (SMOTE, class-weighting) to improve rejection recall
- Hyperparameter tuning (GridSearchCV/Optuna) for Random Forest / Gradient Boosting
- Try XGBoost/LightGBM for potential performance gains
- Calibrate probability outputs if the score will be used for risk-based pricing
