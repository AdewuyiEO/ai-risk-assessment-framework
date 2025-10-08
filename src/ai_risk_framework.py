# AI Risk Assessment & Explainability Framework for Insurance Models

# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import shap
import lime.lime_tabular
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from fairlearn.postprocessing import ThresholdOptimizer
import matplotlib.pyplot as plt

# 2. Data Preparation
# Example: Load and preprocess a sample insurance claims dataset
df = pd.read_csv('data/insurance_claims.csv')
df = df.dropna()
X = df.drop(['claim_approved', 'customer_id'], axis=1)
y = df['claim_approved']

# Encode categorical variables if needed
X = pd.get_dummies(X)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 4. Model Training
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# 5. Baseline Metrics
for model, name in zip([rf, xgb], ['Random Forest', 'XGBoost']):
    y_pred = model.predict(X_test)
    print(f"{name} Classification Report:\n", classification_report(y_test, y_pred))
    print(f"{name} ROC AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.3f}")

# 6. Explainability with SHAP
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test, show=False)
plt.savefig('shap_summary_rf.png')

# 7. Explainability with LIME
lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['Denied', 'Approved'], discretize_continuous=True)
i = 0
exp = lime_explainer.explain_instance(X_test.values[i], rf.predict_proba, num_features=5)
exp.save_to_file('lime_explanation_rf.html')

# 8. Bias/Fairness Testing (Fairlearn)
# Assume 'gender' is a sensitive feature
sensitive = df.loc[X_test.index, 'gender']
mf = MetricFrame(metrics=selection_rate, y_true=y_test, y_pred=rf.predict(X_test), sensitive_features=sensitive)
print("Selection rates by gender:", mf.by_group)
print("Demographic parity difference:", demographic_parity_difference(y_test, rf.predict(X_test), sensitive_features=sensitive))

# 9. Adversarial Robustness (Input Perturbation)
def adversarial_test(model, X, y, perturb_col, epsilon=0.1):
    X_adv = X.copy()
    X_adv[perturb_col] = X_adv[perturb_col] + epsilon
    y_pred_adv = model.predict(X_adv)
    print(f"Adversarial test on {perturb_col}:")
    print(classification_report(y, y_pred_adv))

adversarial_test(rf, X_test, y_test, perturb_col=X_test.columns[0])

# 10. Flask Dashboard (Compliance Report)
from flask import Flask, render_template, send_file
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('report.html')  # Create a simple HTML report

@app.route('/shap_summary')
def shap_summary():
    return send_file('shap_summary_rf.png', mimetype='image/png')

@app.route('/lime_explanation')
def lime_explanation():
    return send_file('lime_explanation_rf.html')

if __name__ == '__main__':
    app.run(debug=True)

# 11. Non-Technical Risk Assessment Report (Template)
"""
AI Model Risk Assessment Report

Business Problem: Predict insurance claim approvals to improve risk scoring and compliance.

Key Findings:
- Model Accuracy: [Insert ROC AUC]
- Explainability: SHAP and LIME highlight key features influencing decisions.
- Bias: Detected demographic parity difference of [Insert Value] for gender.
- Robustness: Model performance degrades by [Insert %] under adversarial input perturbations.

Proposed Controls:
- Regular fairness audits using Fairlearn.
- Human-in-the-loop review for edge cases.
- Monitoring for data drift and retraining triggers.

Conclusion: The model meets baseline performance but requires ongoing governance for fairness and robustness.
"""