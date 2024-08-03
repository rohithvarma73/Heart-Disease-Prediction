import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

def train_and_save_model(csv_file='DATASET.csv', model_file='heart_disease_model.pkl'):
    try:
        print("Loading dataset...")
        # Load the dataset
        data = pd.read_csv(csv_file)
        print("Dataset loaded successfully.")

        # Separate features and target
        X = data.iloc[:, :-1]  # all columns except the last
        y = data.iloc[:, -1]   # the last column

        # Split the data into training and test sets (70% train, 30% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print("Data split into training and test sets.")

        # Initialize and train the logistic regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        print("Model trained successfully.")

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")

        # Save the trained model using pickle
        with open(model_file, 'wb') as file:
            pickle.dump(model, file)
        print("Model saved successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

def load_model(model_file='heart_disease_model.pkl'):
    try:
        print("Loading model...")
        # Load the saved model
        with open(model_file, 'rb') as file:
            model = pickle.load(file)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    train_and_save_model()
