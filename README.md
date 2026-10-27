# Pima Indians Diabetes Diagnostic Classification (Logistic Regression)

> **Supervised clinical machine learning classifier predicting diabetes onset in high-risk populations using Logistic Regression, feature scaling, and diagnostic evaluation.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)

---

## 1. Project Overview

The **Pima Indians Diabetes Classification** project implements a supervised binary classification pipeline to predict the presence of type-2 diabetes based on diagnostic clinical measurements.

Using the widely recognized National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) Pima Indians dataset, this study demonstrates a clean machine learning workflow:
- Exploratory data profiling and zero-value anomaly treatment.
- Continuous feature normalization via `StandardScaler`.
- Probabilistic classification via `LogisticRegression`.
- Diagnostic model evaluation using Confusion Matrices, Precision, Recall, and Classification Reports.

---

## 2. Clinical Feature Schema

All patients in this study are females of Pima Indian heritage aged 21 years and older:

| Feature Name | Description | Units |
| :--- | :--- | :--- |
| `Pregnancies` | Number of times pregnant | Count |
| `Glucose` | Plasma glucose concentration at 2 hours in an oral glucose tolerance test | mg/dL |
| `BloodPressure` | Diastolic blood pressure | mm Hg |
| `SkinThickness` | Triceps skin fold thickness | mm |
| `Insulin` | 2-Hour serum insulin | mu U/ml |
| `BMI` | Body mass index | weight in kg/(height in m)² |
| `DiabetesPedigreeFunction` | Genetic diabetes pedigree score based on family history | Continuous score |
| `Age` | Age in years | Years |
| **`Outcome`** *(Target)* | Class variable (0 = Non-diabetic, 1 = Diabetic) | Binary |

---

## 3. Modeling Workflow

```mermaid
flowchart LR
    A[Raw Clinical Data] --> B[Data Cleaning & Imputation]
    B --> C[Feature Standardization: StandardScaler]
    C --> D[Train / Test Split: 80/20 Stratified]
    D --> E[Logistic Regression Model Training]
    E --> F[Evaluation: Confusion Matrix & ROC]
```

1. **Standardization:** Because features like `Glucose` and `Insulin` operate on drastically different numerical ranges compared to `DiabetesPedigreeFunction`, standard z-score normalization ($z = \frac{x - \mu}{\sigma}$) was applied to ensure uniform gradient updates.
2. **Model Formulation:** Logistic regression models the posterior log-odds:
   $$\log\left(\frac{P(Y=1)}{1 - P(Y=1)}\right) = \beta_0 + \sum_{i=1}^n \beta_i X_i$$
3. **Threshold Evaluation:** The default classification boundary ($p = 0.5$) was evaluated alongside sensitivity-adjusted cutoffs.

---

## 4. Technology Stack

- **Language:** Python 3.10+
- **Machine Learning:** `scikit-learn` (`LogisticRegression`, `StandardScaler`, `train_test_split`, `classification_report`, `confusion_matrix`, `accuracy_score`)
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Environment:** Jupyter Notebook

---

## 5. Repository Structure

```
PIIMA-Indians-Diabetes/
├── Logistic regression dataset (3).ipynb  # End-to-end classification notebook
└── README.md                             # Documentation
```

---

## 6. Setup & Execution

```bash
# 1. Clone repository
git clone https://github.com/keerthi19hub/PIIMA-Indians-Diabetes.git
cd PIIMA-Indians-Diabetes

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# 3. Dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 4. Launch notebook
jupyter notebook
```

---

## 7. Author & Verified Contribution

- **Author:** **Keerthi N** (*Alliance University, BCA*)
- **Verified Contribution:** Implemented data loading, preprocessing, standard scaling, logistic regression modeling, and performance metric evaluation.

---

## 8. License

Released under the [MIT License](https://opensource.org/licenses/MIT).
