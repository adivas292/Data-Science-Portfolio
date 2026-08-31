# Washington State Real Estate Predictive Analytics Engine
### 👤 Developer: Advik Vasanth (UC Irvine Computer Science & Statistics)
🔒 *Production Source Code: PRIVATE (Available for review upon formal interview request)*

## 📐 System Architecture & Logic Flow
This pipeline processes residential property matrices across Washington State to forecast asset valuations. By leveraging mathematical transformations and vectorized feature engineering, the engine stabilizes skewed target data and mitigates dataset noise before compiling predictive boundaries.

```python
# SYSTEM COMPONENT: Vectorized Feature Engineering & Leakage Prevention
# Mapped directly from the private production codebase
def preprocess_washington_housing_data(df):
    # Target Optimization: Resolve severe right-skewness using a log transform
    df["log_price"] = np.log1p(df["price"])
    
    # Outlier Mitigation: Dynamic boundaries using the IQR method to drop anomalies
    Q1 = df["price"].quantile(0.25)
    Q3 = df["price"].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df[~((df["price"] < (Q1 - 1.5 * IQR)) | (df["price"] > (Q3 + 1.5 * IQR)))].copy()
    
    # Vectorized Feature Engineering: Clean data logging errors for house age
    df_clean["house_age"] = np.where(df_clean["yr_renovated"] == 0, 
                                     df_clean["yr_built"], 
                                     df_clean["yr_renovated"])
    
    # Data Leakage Prevention: Strip tracking keys, strings, and derivative inputs
    X = df_clean.drop(columns=["id", "date", "price", "log_price", "price_per_sqft"])
    return X
```

## 📊 Model Performance & Statistical Evaluation
The baseline linear regression engine was trained on an 80% data partition and validated against an unseen 20% testing matrix:

* **R² Score (Coefficient of Determination):** `0.3544` (Physical attributes explain over 35% of total pricing variance)
* **RMSE (Root Mean Squared Error):** `$168,892.58` (Average asset valuation dollar margin of error)
* **Residual Diagnostics:** Prediction errors ($y - \hat{y}$) are randomly distributed above and below the horizontal zero baseline across all pricing tiers, proving homoscedastic stability.

---
© 2026 Advik Vasanth. All rights reserved. Unauthorized cloning, scraping, or distribution of this architectural overview is strictly prohibited.
