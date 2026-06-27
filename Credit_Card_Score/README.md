# 💳 Credit Risk & Default Prediction Dashboard

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

A Machine Learning-powered web application that predicts whether a customer is likely to default on their credit card payments. The application allows users to upload customer datasets, performs preprocessing, predicts risk levels using a trained Logistic Regression model, and generates an interactive dashboard.

---

# 📌 Project Overview

Credit card default prediction helps financial institutions identify customers who are likely to miss repayments. This project analyzes customer financial information and classifies customers into:

* 🟢 Low Risk
* 🔴 High Risk

The project combines Machine Learning with Streamlit to create an easy-to-use interactive dashboard.

---

# 🚀 Features

* 📂 Upload customer dataset (CSV)
* 🤖 Predict customer default risk
* 📈 Interactive dashboard
* 📊 Prediction summary metrics
* 🥧 Customer risk distribution chart
* 📋 Risk assessment table
* 📥 Download prediction report
* 💾 Saved ML model using Joblib

---

# 🧠 Machine Learning Workflow

* Data Collection
* Data Cleaning
* Missing Value Handling
* Feature Encoding
* Exploratory Data Analysis (EDA)
* Correlation Analysis
* Feature Scaling
* Model Training
* Model Evaluation
* Model Deployment using Streamlit

---

# 📊 Models Compared

| Model               | Accuracy |
| ------------------- | -------: |
| Logistic Regression |  **73%** |
| Decision Tree       |      72% |
| Random Forest       |      71% |

### ✅ Final Model Selected

**Logistic Regression**

Reason:

* Highest overall accuracy
* Better precision and recall than the other evaluated models
* Simpler and more interpretable model for credit risk prediction

---

# 📈 Model Performance

| Metric        |               Value |
| ------------- | ------------------: |
| Accuracy      |             **73%** |
| ROC-AUC Score |            **0.66** |
| Features Used |              **85** |
| Algorithm     | Logistic Regression |

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Plotly
* Streamlit
* Joblib

---

# 📷 Application Preview

## Dashboard

![Dashboard](images/dashboard.png)

---

## Prediction Summary

![Prediction Summary](images/prediction_summary.png)

---

## Customer Risk Assessment

![Risk Assessment](images/risk_assessment.png)

---

# 📂 Project Structure

```text
Credit_Card_Default_Prediction/
│
├── app.py
├── Credit_Card_Scoring.ipynb
├── credit_card_model.pkl
├── scaler.pkl
├── credit_score.csv
├── requirements.txt
├── README.md
└── images/
    ├── dashboard.png
    ├── prediction_summary.png
    └── risk_assessment.png
```

---

# ▶️ How to Run

### Clone the repository

```bash
git clone https://github.com/prathama-coder/codealpha_tasks.git
```

### Navigate to the project

```bash
cd Credit_Card_Default_Prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

# 📁 Dataset

The dataset contains customer financial information such as:

* Income
* Savings
* Debt
* Spending Categories
* Financial Ratios
* Credit Score
* Default Status

---

#  Future Improvements

* Hyperparameter tuning
* XGBoost implementation
* Explainable AI using SHAP
* Real-time API integration
* User authentication
* Cloud deployment

---

#  Author

**Prathama Debnath**

B.Tech Computer Science (AI & ML)

SRM Institute of Science and Technology, Ramapuram

GitHub:
https://github.com/prathama-coder

LinkedIn:
https://www.linkedin.com/in/prathama-debnath-467689337

---

#  Acknowledgements

This project was developed as part of the **CodeAlpha Data Science Internship Program** to demonstrate the practical application of Machine Learning in financial risk analysis.
