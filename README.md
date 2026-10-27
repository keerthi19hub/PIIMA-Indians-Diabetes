# 🩺 PIMA Diabetes Diagnostic ML — Diagnostic Classification & Metabolic Risk Predictor

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=881337,E11D48,FDA4AF&height=220&section=header&text=PIMA%20Diabetes%20Diagnostic%20ML&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=Diagnostic%20Classification%20%26%20Metabolic%20Risk%20Predictor&descSize=18&descAlignY=60&descAlign=50" width="100%" alt="PIMA Diabetes Banner" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Algorithm-Logistic%20Regression-881337?style=for-the-badge" alt="Logistic Regression" />
  <img src="https://img.shields.io/badge/Language-Python%203.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Library-Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Tool-Jupyter%20Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
  <img src="https://img.shields.io/badge/Metric-ROC--AUC%20%26%20Precision%20Recall-E11D48?style=for-the-badge" alt="ROC-AUC" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

---

## 📌 Table of Contents

- [📖 Executive Summary](#-executive-summary)
- [🎯 Clinical Diagnostic Context & Objectives](#-clinical-diagnostic-context--objectives)
- [✨ Key Features & Epidemiological Findings](#-key-features--epidemiological-findings)
- [🏗️ Diagnostic Classification Pipeline](#️-diagnostic-classification-pipeline)
- [📊 Model Performance & Diagnostic Metrics](#-model-performance--diagnostic-metrics)
- [🛠️ Tech Stack Matrix](#️-tech-stack-matrix)
- [⚙️ Reproducible Quickstart & Setup](#️-reproducible-quickstart--setup)
- [📁 Project Structure](#-project-structure)
- [👩‍💻 Author & Contributions](#-author--contributions)

---

## 📖 Executive Summary

The **PIMA Indians Diabetes Diagnostic Project** is a clinical machine learning study focused on early diagnostic classification of Type-2 diabetes risk based on physiological biomarkers. Using the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) Pima Indians dataset, the project implements a rigorous diagnostic modeling pipeline utilizing **Logistic Regression**, robust imputation for physiologically implausible zero-values (e.g., glucose, insulin, blood pressure), feature standardization, and Receiver Operating Characteristic (ROC-AUC) curve evaluation.

---

## 🎯 Clinical Diagnostic Context & Objectives

- **Clinical Problem:** Type-2 diabetes often develops asymptomatically over years. Identifying individuals with impaired glucose tolerance early enables dietary intervention before irreversible vascular and neuropathic complications emerge.
- **Modeling Challenges:** Real-world diagnostic data frequently contains biological zero-artifacts (e.g., a measured BMI or diastolic blood pressure of 0 mm Hg, which is clinically impossible in living patients) requiring careful domain-aware imputation rather than naive removal.

---

## ✨ Key Features & Epidemiological Findings

- **🩹 Biological Imputation:** Replaced physiologically impossible zero-values across Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI with median stratified values.
- **📈 Odds Ratio Interpretation:** Leveraged the interpretability of Logistic Regression log-odds coefficients to isolate Glucose concentration and Diabetes Pedigree Function as dominant risk indicators.
- **🎯 Decision Threshold Optimization:** Calibrated probability cutoffs to balance sensitivity (preventing false negatives in high-risk patients) against false alarm rates.

---

## 🏗️ Diagnostic Classification Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 PIMA Diabetes Machine Learning Pipeline                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Clinical Dataset Ingestion (Pregnancies, Glucose, BP, BMI, Pedigree, Age)   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Data Cleansing & Physiological Imputation                                   │
│ - Identify zero-artifacts in biological measurements                        │
│ - Impute non-zero medians across feature distributions                      │
│ - Apply StandardScaler normalization                                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Logistic Regression Modeling & Evaluation                                   │
│ - Compute odds ratios and feature importance weights                        │
│ - Plot Confusion Matrix, ROC-AUC Curve, and Precision-Recall tradeoffs      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Model Performance & Diagnostic Metrics

- **ROC-AUC Score:** ~0.83–0.85
- **Accuracy:** ~78% on held-out test splits
- **Sensitivity / Recall (Class 1):** ~75% after threshold adjustment
- **Clinical Insight:** Glucose level accounts for over 45% of model predictive weight, reinforced by BMI and family pedigree score.

---

## 🛠️ Tech Stack Matrix

| Domain | Technology | Implementation Role |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Primary language for data pipelines and statistical modeling |
| **Machine Learning** | Scikit-Learn | LogisticRegression, train_test_split, StandardScaler, roc_curve |
| **Data Processing** | Pandas, NumPy | Data cleaning, zero-value replacement, exploratory statistics |
| **Visualization** | Seaborn, Matplotlib | Correlation heatmaps, pairplots, ROC curves, confusion matrices |
| **Environment** | Jupyter Notebook | Interactive execution and report documentation |

---

## ⚙️ Reproducible Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/keerthi19hub/PIIMA-Indians-Diabetes.git
cd PIIMA-Indians-Diabetes
```

### 2. Setup Environment & Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate  # Linux/macOS: source venv/bin/activate

pip install pandas numpy scikit-learn seaborn matplotlib jupyter
```

### 3. Run the Diagnostic Notebook
```bash
jupyter notebook "Logistic regression dataset (3).ipynb"
```

---

## 📁 Project Structure

```text
PIIMA-Indians-Diabetes/
├── Logistic regression dataset (3).ipynb  # Primary laboratory notebook with full pipeline
└── README.md                              # Production technical documentation
```

---

## 👩‍💻 Author & Contributions

**Keerthi N**  
- **GitHub:** [@keerthi19hub](https://github.com/keerthi19hub)  
- **Portfolio:** [keerthi19hub.github.io](https://keerthi19hub.github.io)  
- **Course:** Machine Learning & Statistical Diagnostics · Alliance University, Bengaluru  
- **Role:** Clinical data preprocessing, median imputation strategy, Logistic Regression model calibration, and diagnostic metric evaluation.

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=881337,E11D48,FDA4AF&height=100&section=footer" width="100%" alt="Footer Banner" />
</p>
