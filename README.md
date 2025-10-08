# 🧠 AI Risk Assessment & Explainability Framework for Insurance Claim Prediction  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-success)
![ExplainableAI](https://img.shields.io/badge/Explainability-SHAP%20%7C%20LIME-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)


## 📘 Project Overview
This project develops a **robust, explainable, and fair machine learning model** to predict the likelihood of **insurance claims**.  
In the insurance industry, accurate risk assessment is critical for policy pricing, fraud detection, and claim management. However, datasets are often **highly imbalanced**, leading to biased models favoring non-claims.  

This framework integrates:
- **Data balancing (SMOTETomek)**
- **Model optimization (XGBoost)**
- **Explainability (SHAP + feature importance)**
- **Fairness and robustness testing**

It demonstrates how **responsible AI principles**—fairness, transparency, and resilience—can be applied in **insurance governance and compliance**.

---

## Dataset Description

**Dataset Name:** Insurance Claim Prediction Dataset (Kaggle)

### Overview
Historical insurance data containing information about policyholders, claims, and risk factors.

| Category | Description |
|-----------|--------------|
| **Policyholder Info** | Age, gender, marital status, occupation, location |
| **Claim History** | Frequency, amount, type of past claims |
| **Policy Details** | Premium, policy duration, coverage type |
| **Risk Factors** | Credit score, vehicle/health condition |
| **External Factors** | Economic or regional indicators |

**Challenge:** Class imbalance (≈ 93% non-claims vs 7% claims)  
**Solution:** Balanced training data using **SMOTETomek**.

---

## Project Objectives

1. Build a model to predict insurance claims accurately.  
2. Handle class imbalance using advanced resampling techniques.  
3. Evaluate **fairness** across customer demographics.  
4. Apply **explainable AI** tools (SHAP, feature importance).  
5. Conduct **robustness and adversarial** testing.  
6. Create a **compliance-ready AI risk assessment framework**.

---

## Modeling Process

### Stage 1 – EDA & Preprocessing
- Data cleaning, encoding, and feature scaling  
- `ColumnTransformer` for numerical + categorical data  
- SMOTETomek balancing  
- Train/test split (80 / 20)

### Stage 2 – Model Training & Fairness
Compared three classifiers:
| Model | Accuracy | Recall (Claim) | ROC-AUC |
|:------|:----------:|:---------------:|:---------:|
| Logistic Regression | 0.58 | 0.61 | 0.62 |
| Random Forest | 0.88 | 0.09 | 0.59 |
| **XGBoost** | 0.43 | **0.79** | **0.63** |

✅ **Chosen Model:** XGBoost (Balanced + Tuned) — best recall and interpretability for claim prediction.

### Stage 3 – Explainability & Fairness Testing
- Global & local SHAP value interpretation  
- Feature importance ranking  
- Fairness audits across **customer age groups**

---

## Explainability Insights

### Feature Importance (Top 5)
| Rank | Feature | Meaning |
|------|----------|---------|
| 1 | Vehicle Age | Older vehicles → higher claim probability |
| 2 | Customer Age | Younger customers → slightly higher risk |
| 3 | Subscription Length | Shorter policy → higher claim likelihood |
| 4 | Policy Premium | Moderate premiums → stable outcomes |
| 5 | Credit Score | Low scores → increased claim probability |

### SHAP Analysis
- Confirmed **vehicle_age** & **customer_age** as dominant drivers.  
- Local SHAP plots gave interpretable case-level explanations.  

---

## Fairness & Robustness Evaluation

### Fairness by Customer Age Group
| Age Group | ROC-AUC | Observation |
|------------|----------|--------------|
| 18–25 | 0.656 | Good performance |
| 26–35 | 0.626 | Slight drop |
| 36–45 | 0.571 | Lowest; needs feature review |
| 46–60 | 0.627 | Stable |
| 60+ | 0.667 | High but few samples |

### Robustness (Noise Injection)
| Noise Level | ROC-AUC | Δ AUC |
|:------------:|:--------:|:------:|
| 0.01 | 0.554 | −0.08 |
| 0.02 | 0.562 | −0.07 |
| 0.05 | 0.554 | −0.08 |
| 0.10 | 0.551 | −0.08 |

✅ Moderate robustness — performance drop < 10% under noise.

### Adversarial Flip Rates (Top Features)
| Feature | Flip Rate |
|:---------|:-----------:|
| Subscription Length | 0.108 |
| Vehicle Age | 0.320 |
| Customer Age | 0.348 |

---

## Key Takeaways
- **XGBoost + SMOTETomek** gives best recall for rare claim events.  
- **Vehicle Age** and **Customer Age** are core risk drivers.  
- Fairness is largely consistent across groups (ethical readiness ✓).  
- Model shows moderate robustness; can be enhanced via adversarial training.

---
## Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, Scikit-learn, XGBoost, imbalanced-learn, SHAP  
- **Tools:** Jupyter Notebook, Joblib, Matplotlib, Seaborn  
- **Concepts:** Model Fairness, Explainable AI, Adversarial Robustness 

## Repository Structure

## Project Structure
```
insurance-claim-risk/
│
├── data/
│ ├── raw/
│ ├── processed/
│
├── notebooks/
│ ├── 01_EDA_and_Preprocessing.ipynb
│ ├── 02_Model_Training_and_Fairness.ipynb
│ ├── 03_Robustness_and_Adversarial_Tests.ipynb
│
├── artifacts/
│ ├── xgb_balanced_tuned.joblib
│
├── README.md
└── requirements.txt
```
---

## 🏁 Conclusion
This project illustrates a **complete AI risk-assessment pipeline** for insurance — from raw data preprocessing to fairness, explainability, and robustness testing.  
It highlights how combining **machine learning, XAI, and ethical AI** practices can improve both performance and trust in financial models.

---

## 👤 Author
**Adewuyi Elijah Oluwatobi**  
*Data Scientist | AI/ML Engineer | FinTech & Governance Enthusiast*  
📫 [LinkedIn](https://www.linkedin.com/in/elijaholuwatobi) | [GitHub](https://github.com/AdewuyiEO)

