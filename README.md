# Airline Customer Satisfaction Prediction – Logistic Regression

This project analyzes passenger survey data from **Invistico Airline** to predict customer satisfaction (`satisfied` or `dissatisfied`) using **Binomial Logistic Regression**. The analysis covers data preprocessing, handling missing values, encoding categorical features, training/testing split, model evaluation, and interpretation of coefficients to derive actionable business insights.

---

## 📁 Dataset

- **Source**: `Invistico_Airline.csv` (provided)
- **Shape**: 129,880 rows × 22 columns
- **Target**: `satisfaction` (satisfied / dissatisfied)
- **Features** include:
  - Numerical: Age, Flight Distance, service ratings (Seat comfort, Inflight entertainment, etc.), departure/arrival delays
  - Categorical: Customer Type, Type of Travel, Class

---

## 🔧 Workflow

1. **Load & Inspect** – Load CSV, check missing values.
2. **Handle Missing Values** – Convert delay columns to numeric, fill with median; drop rows missing target.
3. **Encode Target** – Create binary `satisfied` (1 = satisfied, 0 = dissatisfied).
4. **Split Features** – Separate features (`X`) and target (`y`).
5. **Preprocessing Pipeline**:
   - Numerical features: StandardScaler
   - Categorical features: OneHotEncoder (drop first)
6. **Train/Test Split** – 80/20 stratified split.
7. **Logistic Regression** – `max_iter=1000`, `random_state=42`.
8. **Evaluation** – Confusion matrix, classification report, ROC curve, AUC.
9. **Interpretation** – Extract coefficients and odds ratios to identify drivers of satisfaction.

---

## 📈 Model Performance

- **Accuracy**: 82.90%
- **Precision (satisfied)**: 84.41%
- **Recall (satisfied)**: 84.33%
- **ROC AUC**: 0.9035

**Confusion Matrix** (test set: 25,976 samples)

|               | Predicted Dissatisfied | Predicted Satisfied |
|---------------|------------------------|---------------------|
| Actual Dissatisfied | 9,544                  | 2,215               |
| Actual Satisfied    | 2,228                  | 11,989              |

**Classification Report**

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| Dissatisfied | 0.81      | 0.81   | 0.81     | 11,759  |
| Satisfied    | 0.84      | 0.84   | 0.84     | 14,217  |

---

## 🔍 Key Drivers of Satisfaction

### Top Positive Drivers (increase satisfaction)

| Feature                 | Coefficient | Odds Ratio |
|-------------------------|-------------|------------|
| Inflight entertainment  | 0.973       | 2.647       |
| Seat comfort            | 0.391       | 1.479       |
| On-board service        | 0.387       | 1.473       |
| Checkin service         | 0.355       | 1.426       |
| Ease of Online booking  | 0.334       | 1.396       |

### Top Negative Drivers (decrease satisfaction)

| Feature                         | Coefficient | Odds Ratio |
|---------------------------------|-------------|------------|
| Customer Type (disloyal)        | -1.874      | 0.153       |
| Class (Eco Plus)                | -0.764      | 0.466       |
| Type of Travel (Personal Travel)| -0.759      | 0.468       |
| Class (Eco)                     | -0.712      | 0.491       |
| Departure/Arrival time convenient | -0.327    | 0.721       |

> **Interpretation**:  
> - A one‑unit increase in **Inflight entertainment** rating multiplies the odds of satisfaction by **2.65**.  
> - Being a **disloyal customer** reduces the odds of satisfaction by about **85%** compared to loyal customers.  
> - **Personal travel** and flying **Economy/Economy Plus** are strongly associated with lower satisfaction.

---

## 🧪 How to Run

### Requirements

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Execution

1. Place `Invistico_Airline.csv` in the same directory as the notebook.
2. Run all cells in `Logistic_Regression_Evaluation.ipynb`.

The notebook will output:
- Dataset shape and missing value summary
- Train/test sizes
- Confusion matrix and classification report
- ROC curve plot
- Coefficient table with top positive/negative drivers

---

## 📊 Visualization

The ROC curve shows an AUC of **0.9035**, indicating excellent discriminative ability.

![ROC Curve](roc_curve.png)

*(Generated in notebook)*

---

## 📌 Conclusions

- Logistic regression achieves **83% accuracy** and **0.90 AUC** – a strong baseline for binary satisfaction prediction.
- **Inflight entertainment**, **seat comfort**, and **on‑board service** are the most influential features driving satisfaction.
- **Disloyal customers**, **personal travel**, and **economy classes** are major risk factors for dissatisfaction.
- Actionable recommendations: Invest in entertainment and seat comfort, target loyalty programmes, and improve service for economy/personal travel segments.

---

## 📄 License

This project is for educational/demonstration purposes only. Dataset belongs to Invistico Airline (simulated survey data).
