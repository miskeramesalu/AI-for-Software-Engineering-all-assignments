# AI Development Workflow Assignment
# Part 1: Short Answer Questions (30 points)

## 1. Problem Definition (6 points)

**Problem statement:** Predict student dropout risk for the next academic semester using student records, engagement, and socio-economic data.

**Objectives:**

1. Identify students at high risk of dropping out early enough for intervention.
2. Determine key predictors of dropout risk for academic counseling.
3. Reduce dropout rates by 10% within one academic year.

**Stakeholders:**

* University administration and academic counselors.
* Students and their families.

**Key Performance Indicator (KPI):**

* **Recall@K** — proportion of students who dropped out that were correctly flagged among the top K% of predictions.

---

## 2. Data Collection & Preprocessing (8 points)

**Data sources:**

1. Student Information System (demographics, grades, attendance).
2. Learning Management System (LMS) activity logs, login frequency, and assignment submissions.

**Potential bias:**

* **Selection bias** — students with limited internet access may have fewer LMS interactions, leading to overestimation of dropout risk.

**Preprocessing steps:**

1. **Missing value handling:** Impute grades with course means or model-based estimates; flag missing demographics.
2. **Normalization:** Use StandardScaler to normalize continuous features like GPA and attendance.
3. **Encoding:** Apply one-hot encoding to categorical variables such as program or major.

---

## 3. Model Development (8 points)

**Model:** Gradient Boosted Trees (XGBoost or LightGBM).
**Justification:** Handles mixed feature types, robust to missing values, interpretable using SHAP values.

**Data split:** 70% train, 15% validation, 15% test (time-based split to prevent leakage).

**Hyperparameters:**

1. **Learning rate:** Controls model convergence and generalization.
2. **Max depth:** Prevents overfitting by controlling tree complexity.

---

## 4. Evaluation & Deployment (8 points)

**Evaluation metrics:**

1. **Recall:** Measures ability to correctly identify high-risk students.
2. **Precision@K:** Ensures flagged students are truly at risk, minimizing wasted interventions.

**Concept drift:**

* **Definition:** Change in feature-target relationships over time.
* **Monitoring:** Track performance metrics and feature distributions periodically; retrain when drift exceeds threshold.

**Deployment challenge:**

* **Scalability:** Ensuring low-latency scoring for thousands of students daily. Mitigate via model serving frameworks (TensorFlow Serving, FastAPI + Docker).

---

# Part 2: Case Study — Hospital Readmission Prediction (40 points)

**Scenario:** Predict which patients will be readmitted within 30 days post-discharge.

## 1. Problem Scope (5 points)

**Objectives:**

1. Predict readmission risk for each discharged patient.
2. Explain risk factors to clinicians.
3. Reduce unnecessary readmissions and associated costs.

**Stakeholders:**

* Clinicians and case managers.
* Patients and families.
* Hospital administrators and insurance providers.

---

## 2. Data Strategy (10 points)

**Data sources:**

1. **EHRs:** Diagnoses, procedures, medications, lab results.
2. **Demographics:** Age, gender, socioeconomic data.
3. **Claims:** Previous admissions and emergency visits.

**Ethical concerns:**

1. Patient privacy — ensure de-identification and encryption.
2. Bias — underrepresented groups may face inaccurate predictions.

**Preprocessing pipeline:**

1. **De-identify** patient data.
2. **Handle missing values** using medical domain rules.
3. **Aggregate** clinical data over a 90-day window.
4. **Feature engineering:** comorbidity scores, medication counts, visit frequency.
5. **Encoding/scaling:** Normalize continuous variables, one-hot encode categorical ones.
6. **Label:** Binary — readmitted within 30 days = 1.

---

## 3. Model Development (10 points)

**Model:** LightGBM (preferred for tabular health data).

**Confusion matrix (hypothetical):**

```
               Predicted Positive   Predicted Negative
Actual Positive       80                 40
Actual Negative       120                760
```

**Precision = 80 / (80 + 120) = 40%**
**Recall = 80 / (80 + 40) = 66.7%**

Interpretation: The model correctly identifies two-thirds of at-risk patients but produces some false positives.

---

## 4. Deployment (10 points)

**Integration steps:**

1. Build a REST API endpoint for predictions.
2. Automate data ingestion from EHR.
3. Authenticate users (clinicians only).
4. Embed outputs in clinician dashboards.
5. Log predictions and retrain periodically.

**Regulatory compliance:**

* Encrypt data in transit and at rest.
* Use HIPAA-compliant storage.
* Maintain access logs and audit trails.

---

## 5. Optimization (5 points)

**Overfitting control:**

* Apply **k-fold cross-validation**, early stopping, and L2 regularization.
* Limit model depth and prune low-importance features.

---

# Part 3: Critical Thinking (20 points)

## 1. Ethics & Bias (10 points)

**Effect of bias:** Skewed training data may cause unequal care — underpredicting risk for certain groups and reinforcing disparities.

**Mitigation strategy:** Audit subgroup performance, reweight data, or use fairness-aware post-processing (Equalized Odds, demographic parity).

---

## 2. Trade-offs (10 points)

**Interpretability vs. accuracy:** In healthcare, explainability is crucial. A slightly less accurate model (e.g., logistic regression) may be chosen over black-box models for clinician trust and compliance.

**Limited resources:** Choose lightweight models; precompute features; perform batch scoring. Trade minimal accuracy loss for operational efficiency.

---

# Part 4: Reflection & Workflow Diagram (10 points)

## 1. Reflection (5 points)

**Most challenging:** Data cleaning and ensuring privacy — healthcare data is incomplete, inconsistent, and privacy-restricted.

**Improvements:** With more resources, integrate real-time EHR data, add interpretability dashboards, and test across departments.

---

## 2. Workflow Diagram (5 points)

**Flow:**

1. Problem Definition → 2. Data Collection → 3. Preprocessing → 4. EDA → 5. Modeling → 6. Evaluation → 7. Deployment → 8. Monitoring → 9. Feedback & Retraining.

Use draw.io or Lucidchart to create this flow with decision arrows (e.g., if model fails → return to feature engineering).

# References

* CRISP-DM Framework
* LightGBM/XGBoost Docs
* WHO Guidelines on Data Ethics
* HIPAA Privacy Rule Documentation

# Layout
ai_readmission_project/
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── data_processing.py
│   ├── features.py
│   ├── train.py
│   └── serve.py
├── models/
├── outputs/
│   ├── charts/
│   └── logs/
├── requirements.txt
└── docs/
    └── answers( part 1 up to part 4)
