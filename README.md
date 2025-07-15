# 🫀 AI-Powered Heart Disease Prediction

An intelligent web application that predicts the likelihood of heart disease in patients using machine learning. Built using **Flask**, **PyTorch**, and **Scikit-learn**, the model analyzes key medical parameters to provide a quick and accessible risk assessment.

## 🚀 Features

- 🔍 Predicts heart disease risk using key health indicators
- 📈 Trained with supervised machine learning algorithms (Logistic Regression, Random Forest, etc.)
- 💡 Optionally powered by a deep learning model using PyTorch
- 🌐 Simple and responsive web interface using Flask
- 📊 Visualizations for EDA and model performance

## 🧠 Tech Stack

- **Frontend**: HTML5, CSS3, Bootstrap
- **Backend**: Python, Flask
- **ML/DL**: Scikit-learn, PyTorch
- **Data**: UCI Heart Disease Dataset
- **Others**: Pandas, NumPy, Matplotlib, Seaborn

## 📝 Dataset

The model is trained on the [UCI Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease), which contains 300+ rows and 14 clinical features such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Serum Cholesterol
- Fasting Blood Sugar
- Max Heart Rate Achieved
- ST Depression

## 🧪 Model Training

- Data preprocessing and feature scaling
- Exploratory Data Analysis (EDA)
- Trained using multiple ML algorithms
- Evaluated with accuracy, precision, recall, and ROC-AUC
- Deployed best-performing model (e.g., Logistic Regression or PyTorch NN)

## 🖥️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/heart-disease-predictor.git
cd heart-disease-predictor

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
