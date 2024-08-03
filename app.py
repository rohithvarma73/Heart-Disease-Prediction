from flask import Flask, request, render_template
import numpy as np
from backend import load_model

# Load the saved model
model = load_model()

# Create a Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['chest_pain_type']),
            float(request.form['bp']),
            float(request.form['cholesterol']),
            float(request.form['fbs']),
            float(request.form['ekg_results']),
            float(request.form['max_hr']),
            float(request.form['exercise_angina']),
            float(request.form['st_depression']),
            float(request.form['slope_of_st']),
            float(request.form['num_vessels_fluro']),
            float(request.form['thallium'])
        ]
        final_features = [np.array(features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        
        # Get prediction result
        output = 'Presence' if prediction[0] == 1 else 'Absence'
        
        return render_template('index.html', prediction_text=f'Heart Disease Prediction: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
