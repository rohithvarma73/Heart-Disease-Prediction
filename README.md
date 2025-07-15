# 🫀 AI-Powered Heart Disease Predictor

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent AI-based web application that predicts the likelihood of **heart disease** based on clinical parameters using **machine learning** and **deep learning** techniques. Built using **Flask**, **PyTorch**, and **Scikit-learn**, this project provides a lightweight and interactive user interface for real-time risk assessment.

---

## 🚀 Features

- 🧠 Predicts heart disease risk using trained ML and DL models
- 📉 Trained on real-world UCI dataset
- ⚙️ REST API backend with Flask
- 📊 EDA and model metrics included
- 🌐 Responsive web UI (Bootstrap)
- ☁️ Optionally deployed on **Render**

---

## 🧰 Tech Stack

| Area        | Tools / Libraries |
|-------------|-------------------|
| Frontend    | HTML, CSS, Bootstrap |
| Backend     | Python, Flask     |
| ML / DL     | Scikit-learn, PyTorch |
| Visualization | Matplotlib, Seaborn |
| Dataset     | UCI Heart Disease Dataset |
| Deployment  | Render (free tier) |

---

## 📝 Dataset

Used the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease), which includes 14 features:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- ECG Results
- Max Heart Rate
- Exercise-Induced Angina
- ST Depression
- Slope of Peak
- Number of Major Vessels
- Thalassemia
- Target (0 = No Disease, 1 = Disease)

---

## 🧠 Model Training

- Handled missing values and encoded categorical data
- Scaled numeric features with `StandardScaler`
- Trained and compared models: Logistic Regression, Random Forest, PyTorch Neural Network
- Evaluated using:
  - Accuracy
  - Precision & Recall
  - ROC-AUC Score

---

## 🖥️ Running the App Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/heart-disease-predictor.git
cd heart-disease-predictor

# (Optional) Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
