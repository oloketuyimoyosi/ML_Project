# 🏦 Loan Prediction System

> **Status:** Development
> **Python Version:** 3.8+

## 📖 Overview
This project is a Machine Learning application designed to automate the loan eligibility process. It predicts whether a loan applicant should be **Approved** or **Rejected** based on specific criteria such as income, education, credit history, and property area.

**Goal:** To help the team understand how the model makes decisions and to provide a simple interface for testing predictions.

### 🧠 The Model
* **Algorithm:** [e.g., Logistic Regression / Random Forest Classifier]
* **Key Predictors:** Credit History (most important), Applicant Income, Loan Amount.
* **Dataset:** Lending Club loan dataset
* **Accuracy:** Currently achieving ~80% accuracy on the validation set.

---

## 🛠️ Tech Stack
* **Core:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Visualization:** Matplotlib / Seaborn
* **Interface:** [e.g., Streamlit / Flask] (Used for the web dashboard)

---

## 🚀 Getting Started

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository
```bash
git clone <repository_url>
cd loan-prediction-app
### 2.Create a Virtual Environment (Recommended)

streamlit run app.py
### 3. Install Dependencies
pip install -r requirements.txt
### 4. Run 
streamlit run app.py

### How it works 

## Test Case 1: The "Ideal" Borrower (Should be LOW Risk)
Use these values to test if your model correctly identifies a safe loan.
Loan Amount: $5,000
Term: 36 Months
Interest Rate: 5.5%
Installment: $150
Grade: A
Annual Income: $120,000
DTI (Debt-to-Income): 5.0
Home Ownership: MORTGAGE
Employment Length: 10+ years
Expected Result: Low Probability (< 10%)
Why? High income, low loan amount, and excellent credit grade usually mean the borrower will pay back.