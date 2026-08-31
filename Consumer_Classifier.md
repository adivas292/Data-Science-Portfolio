# Indian Consumer Credit Behavior Classification Model
[← Back to Portfolio Home](./README.md)

## 📊 Project Overview
This data science project constructs a binary logistic regression classifier using `scikit-learn` to predict consumer demographic profiles based on credit card transaction traits. After proving via exploratory data diagnostics that raw transaction amounts possessed zero linear correlation with consumer features (R²: -0.0001), the pipeline was refactored from a regression baseline into a classification framework to isolate subtle behavioral spending signals across 26,052 real-world retail transactions.

---

## 🛠️ Key Technical Implementations
* **Pipeline Refactoring**: Successfully pivoted a dead-end regression framework into a binary logistic classification system, shifting the target variable from continuous pricing rows to demographic indicators.
* **Chronological Feature Engineering**: Parsed raw calendar text strings to extract day-of-the-week indexes, engineering a custom behavioral metric (`is_weekend`) to isolate shifting consumer weekend expenditure patterns.
* **Categorical Matrix Encoding**: Implemented multi-column One-Hot Encoding via Pandas across dimensional transaction factors (`Card Type` and `Expense Categories`), applying explicit integer type casting (`.astype(int)`) to eliminate boolean noise and guarantee matrix shape alignment.
* **Data Leakage Prevention**: Explicitly dropped unique system constants, text strings, tracking identifiers, and regional city inputs from the final feature grid (X), maintaining strict predictive validation integrity.

---

## 📐 Production Code Architecture (Logic Blueprint)
```python
# Mapped directly from the private production codebase
import pandas as pd
import numpy as np

def preprocess_credit_behavior_data(df):
    # Chronological Feature Engineering: Extract weekend behavioral indicators
    df["purchase_date"] = pd.to_datetime(df["Date"])
    df["is_weekend"] = df["purchase_date"].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Categorical Matrix Encoding: Execute One-Hot Encoding across dimensional vectors
    categorical_cols = ["Card Type", "Exp Type"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Cast boolean dummies to explicit integers to eliminate matrix noise
    dummy_columns = [col for col in df_encoded.columns if any(p in col for p in categorical_cols)]
    df_encoded[dummy_columns] = df_encoded[dummy_columns].astype(int)
    
    # Data Leakage Prevention: Drop system constants, tracking IDs, and target strings
    # Target variable 'Gender' is isolated separately for the y vector
    X = df_encoded.drop(columns=["index", "Date", "purchase_date", "City", "Amount", "Gender"])
    return X
```

## 📈 Model Performance & Statistical Evaluation
The logistic classifier was fitted to an 80% training data partition and validated against an unseen 20% testing matrix containing 5,211 independent transaction records:

* **Overall Classification Accuracy:** `53.60%` (Successfully outperforming a random 50/50 blind coin-flip threshold)
* **Dataset Baseline Integration:** Proved raw transaction amounts possessed zero linear correlation with consumer attributes (\(R^2\): -0.0001), justifying the architectural pivot into behavioral classification models.

### 📊 Confusion Matrix Diagnostics
Using automated `ConfusionMatrixDisplay` pipelines, validation predictions were bucketed into a strict 2x2 grid to evaluate underlying classification behavior:
* **True Negatives (Correctly Identified Males):** `733`
* **True Positives (Correctly Identified Females):** `2,060`
* **False Negatives (Missed Females):** `694`
* **False Positives (Missed Males):** `1,724`

**Analytical Takeaway:** The heavy concentration of True Positives and False Positives highlights a distinct classification bias toward the majority class in the training dataset. This visual diagnosis provides a direct avenue for future production optimization via synthetic oversampling (SMOTE) or the integration of continuous regional parameters.

---
© 2026 Advik Vasanth. All rights reserved.
