import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Global variables to cache loaded models
_model = None
_scaler = None

def train_model():
    """Train the model and save it. Run this ONLY when you need to retrain."""
    print("Training model...")
    df = pd.read_excel('diabetes.xls')
    x = df.drop(columns='Outcome', axis=1)
    y = df['Outcome']

    data_scaling = StandardScaler()
    data_scaling.fit(x)
    standardized_data = data_scaling.transform(x)
    x = standardized_data

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    model = RandomForestClassifier(random_state=1)
    model.fit(x_train, y_train)

    # Save the model and scaler
    joblib.dump(model, 'diabetes_model.joblib')
    joblib.dump(data_scaling, 'diabetes_scaler.joblib')
    print("Model trained and saved!")

def load_model():
    """Load the pre-trained model and scaler. Cache in memory."""
    global _model, _scaler
    
    if _model is None or _scaler is None:
        try:
            print("Loading pre-trained model...")
            _model = joblib.load('diabetes_model.joblib')
            _scaler = joblib.load('diabetes_scaler.joblib')
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Model files not found. Training new model...")
            train_model()
            _model = joblib.load('diabetes_model.joblib')
            _scaler = joblib.load('diabetes_scaler.joblib')
    
    return _model, _scaler

def predict_diabetes(data):
    """Make prediction using cached model."""
    # Import pandas only when needed
    import pandas as pd
    
    model, scaler = load_model()

    input_data_df = pd.DataFrame([data], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])

    std_data = scaler.transform(input_data_df)
    prediction = model.predict(std_data)

    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"

# REMOVED THIS LINE - No more training on startup!
# train_model()  # <-- This was causing the slow startup!