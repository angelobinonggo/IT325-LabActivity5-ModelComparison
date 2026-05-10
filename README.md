# IT325 Machine Learning — Lab Activity 5
## Training and Comparing Five Classification Models

| | |
|---|---|
| **Name** | John Michael Angelo C. Binonggo |
| **Program & Section** | BSIT-3R12 |
| **Course** | IT325 Machine Learning |
| **Activity** | Laboratory Activity 5 — Model Comparison |

---

## Overview

This repository contains the Jupyter Notebook for **Lab Activity 5** of IT325 Machine Learning. The goal of this activity is to train and evaluate **five classification models** on the same dataset split, compare their performance using standard evaluation metrics, and identify the top two candidates for hyperparameter tuning.

---

## Repository Contents

```
├── lab_activity5_compare_5_models.ipynb   # Main Jupyter Notebook
├── asthma_disease_data.csv                # Dataset used for model training
└── README.md                              # This file
```

---

## Dataset

- **File:** `asthma_disease_data.csv`
- **Records:** 2,392 patients
- **Features:** 26 clinical and demographic features
- **Target:** `Diagnosis` (0 = No Asthma, 1 = Asthma)
- **Class Distribution:** ~94.8% No Asthma / ~5.2% Asthma *(severely imbalanced)*

---

## Models Trained

| # | Model |
|---|-------|
| 1 | Logistic Regression |
| 2 | K-Nearest Neighbors (KNN) |
| 3 | Decision Tree |
| 4 | Random Forest |
| 5 | Naive Bayes |

---

## Methodology

1. **Data Preparation** — Dropped non-informative columns (`PatientID`, `DoctorInCharge`)
2. **Train-Test Split** — 80% training / 20% testing (`random_state=42`, `stratify=y`)
3. **Feature Scaling** — `StandardScaler` fit on training set only (prevents data leakage)
4. **Class Imbalance Handling** — SMOTE applied on training set only to balance classes (1,814 vs 1,814 samples after resampling)
5. **Model Training** — All five models trained on the same SMOTE-resampled training set
6. **Evaluation** — Accuracy, Precision, Recall, F1-Score, and Confusion Matrix on the original test set

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.5804 | 0.0686 | 0.5600 | 0.1223 |
| K-Nearest Neighbors | 0.5720 | 0.0455 | 0.3600 | 0.0807 |
| Decision Tree | 0.8768 | 0.0526 | 0.0800 | 0.0635 |
| Random Forest | 0.9478 | 0.0000 | 0.0000 | 0.0000 |
| **Naive Bayes** | 0.7056 | **0.0972** | **0.5600** | **0.1657** ✅ |

> **Note:** Due to the extreme class imbalance (94.8% majority class), raw accuracy is not a reliable metric. **Recall and F1-Score for the Asthma (minority) class** are the primary metrics for model selection.

---

## Top Two Models Selected for Tuning

1. **Naive Bayes** — Best F1-Score (0.1657) and Recall (0.56) for the Asthma class
2. **Logistic Regression** — Second-best F1-Score (0.1223) with Recall of 0.56, plus strong interpretability for medical use cases

---

## Notebook Link

> *(Google Colab link to be added after upload)*

---

## Requirements

```
scikit-learn
imbalanced-learn
pandas
numpy
matplotlib
seaborn
```

Install with:
```bash
pip install scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
```
