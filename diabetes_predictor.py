import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
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

def predict_diabetes(data):
    model = joblib.load('diabetes_model.joblib')
    scaler = joblib.load('diabetes_scaler.joblib')

    input_data_df = pd.DataFrame([data], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])

    std_data = scaler.transform(input_data_df)
    prediction = model.predict(std_data)

    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"

# Train and save the model (run this once)
train_model()