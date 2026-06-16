# Student Dropout Risk Prediction using TensorFlow Neural Network

## Overview

Student Dropout Risk Prediction is a machine learning project that predicts whether a student is at risk of dropping out or graduating based on academic, demographic, financial, and enrollment-related information.

The project uses the **Predict Students' Dropout and Academic Success** dataset from the UCI Machine Learning Repository and implements a complete machine learning workflow, including data preprocessing, exploratory data analysis, model development, evaluation, and deployment through an interactive Streamlit dashboard.

The goal is to demonstrate how artificial intelligence can support educational institutions in identifying students who may require early academic or financial intervention.

---

## Dataset

**Dataset:** Predict Students' Dropout and Academic Success

**Source:** UCI Machine Learning Repository

The dataset contains information about students' academic performance, enrollment characteristics, socio-economic background, and institutional factors.

### Original Target Classes

* Dropout
* Enrolled
* Graduate

### Binary Classification Target

For this project, the problem was simplified into a binary classification task:

| Original Class | Encoded Target |
| -------------- | -------------- |
| Graduate       | 0              |
| Dropout        | 1              |

The "Enrolled" class was excluded to focus on distinguishing students who successfully graduated from those who dropped out.

---

## Project Objectives

* Predict student dropout risk using machine learning and deep learning techniques.
* Compare traditional machine learning algorithms with a neural network model.
* Identify factors associated with student dropout.
* Build an interactive dashboard for real-time prediction.
* Demonstrate an end-to-end AI workflow suitable for educational technology applications.

---

## Project Workflow

### 1. Data Understanding

* Dataset exploration
* Feature inspection
* Missing value analysis
* Class distribution analysis

### 2. Exploratory Data Analysis (EDA)

Several analyses were conducted to understand relationships between features and dropout risk:

* Target distribution
* Admission grade analysis
* Age at enrollment analysis
* Scholarship status analysis
* Tuition fee status analysis
* Semester performance analysis
* Correlation analysis

### 3. Data Preprocessing

* Target filtering and encoding
* Train-test split
* Feature scaling using StandardScaler
* Class imbalance handling using class weights

### 4. Model Development

Three models were developed and compared:

#### Logistic Regression

Used as a baseline machine learning model.

#### Random Forest

Used as a stronger baseline model and for feature importance analysis.

#### TensorFlow Neural Network

Main deep learning model consisting of:

* Dense Layer (64 neurons)
* Dropout Layer
* Dense Layer (32 neurons)
* Dropout Layer
* Output Layer (Sigmoid)

---

## Model Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

Recall was considered particularly important because correctly identifying students at risk of dropout is more valuable than maximizing overall accuracy alone.

---

## Feature Importance

Feature importance analysis was performed using the Random Forest model.

Examples of influential features include:

* Curricular units 2nd semester (approved)
* Curricular units 2nd semester (grade)
* Curricular units 1st semester (approved)
* Curricular units 1st semester (grade)
* Tuition fees up to date
* Scholarship holder
* Debtor status
* Admission grade
* Age at enrollment

These factors provide useful insights into student success and dropout patterns.

---

## Streamlit Dashboard

The project includes an interactive Streamlit application that allows users to:

### Manual Prediction

Users can manually enter student information and receive:

* Dropout probability
* Predicted class
* Risk level
* Recommendation

### Batch Prediction

Users can upload a CSV file and generate predictions for multiple students simultaneously.

### Feature Guide

The dashboard provides explanations for encoded categorical values to improve usability for non-technical users.

---

## Risk Classification

Predicted probabilities are converted into risk categories:

| Probability | Risk Level  |
| ----------- | ----------- |
| 0.00 – 0.39 | Low Risk    |
| 0.40 – 0.69 | Medium Risk |
| 0.70 – 1.00 | High Risk   |

---

## Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Streamlit
* Joblib

---

## Project Structure

```text
student-dropout-risk-prediction/
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   ├── dropout_prediction_model.keras
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── feature_info.pkl
│
├── notebook/
│   └── student_dropout_prediction.ipynb
│
├── data/
│   └── data.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Underag3/student-dropout-risk-prediction.git
cd student-dropout-risk-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

---

## Future Improvements

Potential enhancements for future development:

* Hyperparameter tuning
* SHAP explainability analysis
* Advanced neural network architectures
* Cloud deployment
* Real-time API integration
* Multi-class classification (Dropout, Enrolled, Graduate)

---

## Author

**Mohammad Tyas Subianto**

Data Analyst | Machine Learning Enthusiast | AI & Data Science Student

GitHub: https://github.com/Underag3

LinkedIn: https://www.linkedin.com/in/m-tyas-subianto

---

## Acknowledgements

This project uses the Predict Students' Dropout and Academic Success dataset provided by the UCI Machine Learning Repository and is intended for educational and portfolio purposes.
