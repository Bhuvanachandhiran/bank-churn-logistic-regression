# Bank Customer Churn Prediction using Logistic Regression  

**Portfolio Project • Python • Scikit-learn • EDA • Interpretable ML**

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/bank-churn-logistic-regression/blob/main/Bank_Churn_Prediction_Logistic_Regression.ipynb)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/competitions/playground-series-s4e1)

---

### Project Overview
End-to-end machine learning project that predicts **whether a bank customer will churn (Exited = 1)** using **Logistic Regression** — the gold standard for interpretable classification in finance.

Dataset: **Kaggle Playground Series S4E1 (2024)** – 165,034 real-world-like customer records.

---

### Key Results
- **ROC-AUC ≈ 0.85** on hold-out test set (excellent for logistic regression)
- Handled **class imbalance** (21% churn) with **SMOTE**
- Full preprocessing: encoding, scaling, feature alignment
- Interpretable coefficients + odds ratios
- Top churn drivers identified: **Germany**, **older age**, **low activity**, **3–4 products**
- Saved model + prediction function ready for deployment
- Kaggle submission file included (public leaderboard ready)

---

### Dataset
- Source: [Kaggle Playground Series - Season 4, Episode 1](https://www.kaggle.com/competitions/playground-series-s4e1/data)
- 165,034 training rows • 14 features • Binary target (`Exited`)
- Features: `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`

---

### Project Structure
bank-churn-logistic-regression/
├── Bank_Churn_Prediction_Logistic_Regression.ipynb   # Main notebook (Colab-ready)
├── train.csv                                          # Auto-downloaded via Kaggle API
├── test.csv                                           # For Kaggle submission
├── submission.csv                                     # Final predictions
├── churn_logreg_model.pkl                             # Trained model
├── scaler.pkl                                         # Fitted scaler
├── requirements.txt
└── README.md                                          # This file

---

### How to Run (2 Minutes – No Setup Needed)
1. Click **"Open in Colab"** badge above  
2. Run the first cell → upload your `kaggle.json` (one-time only)  
3. Run all cells → dataset downloads & model trains automatically!
