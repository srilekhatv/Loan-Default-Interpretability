# 🧠 FairLend: Interpretable & Fair Loan Default Prediction

This project builds a machine learning pipeline to **predict loan default** while ensuring fairness, transparency, and explainability — aligned with ethical AI practices.

## 📌 Objective
Develop a binary classification model to predict whether a loan applicant is likely to default. The project emphasizes **interpretable and fair ML**, not just performance.

## 🧰 Tools & Libraries
- Python, Pandas, Scikit-learn
- Random Forest Classifier
- SHAP, LIME, DiCE-ML
- AIF360, Fairlearn
- PyALE, PDPBox, InterpretML

## 🧪 Pipeline Overview
1. **Data Cleaning & Preprocessing**  
   - Handled missing values, encoded categorical variables, and scaled numerical features.
2. **Handling Class Imbalance**  
   - Applied **SMOTE** to upsample the minority (default) class.
3. **Model Training**  
   - Trained a **Random Forest Classifier** on SMOTE-balanced data.
4. **Evaluation Metrics**  
   - Accuracy, Precision, Recall, F1-score, and ROC-AUC across train/test/unseen sets.
5. **Interpretability & Explainability**
   - Used **SHAP**, **LIME**, and **DiCE** to generate:
     - Global explanations (feature importance, SHAP summary)
     - Local explanations (LIME + SHAP force plots)
     - Counterfactuals (DiCE) and anchors (Alibi)
6. **Fairness Assessment**
   - Evaluated bias using **AIF360** and **Fairlearn** with metrics like:
     - Demographic Parity Difference
     - Equalized Odds Difference
   - Analyzed disparities across features like income, gender, and location.
7. **Visualizations**
   - Used **ALE**, **PDP**, and **SHAP** plots to visualize feature influence.

## ✅ Results
- Achieved high recall and AUC on default class after SMOTE.
- SHAP revealed top predictors: income, job tenure, and car ownership.
- Fairness metrics showed measurable bias across some features, prompting discussion on mitigation strategies.

## 📂 Folder Structure
```
Loan_Default_Interpretability/
│
├── Loan_Default_Interpretability.ipynb
├── README.md
|__ Dataset
```

## 📌 Run Instructions
```bash
# Clone the repo
git clone https://github.com/srilekhatv/Loan_Default_Interpretability.git
cd Loan_Default_Interpretability

# Open the notebook in Jupyter or Colab
```

## ✍️ Author
Srilekha Tirumala Vinjamoori  
[LinkedIn](https://www.linkedin.com/in/srilekha-tirumala-vinjamoori/) • [Portfolio](https://srilekhatv-portfolio.vercel.app)
