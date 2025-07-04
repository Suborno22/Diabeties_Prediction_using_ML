# train_model.py - Create this file and run it ONCE locally
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_and_save_model():
    print("Loading data...")
    df = pd.read_excel('diabetes.xls')
    
    print("Preparing data...")
    x = df.drop(columns='Outcome', axis=1)
    y = df['Outcome']

    # Standardize the features
    data_scaling = StandardScaler()
    data_scaling.fit(x)
    standardized_data = data_scaling.transform(x)
    x = standardized_data

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    print("Training model...")
    model = RandomForestClassifier(random_state=1)
    model.fit(x_train, y_train)

    # Test the model
    y_test_pred = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Data Accuracy: {test_accuracy:.2f}")

    # Save the model and scaler
    print("Saving model and scaler...")
    joblib.dump(model, 'diabetes_model.joblib')
    joblib.dump(data_scaling, 'diabetes_scaler.joblib')
    
    print("✅ Model trained and saved successfully!")
    print("Files created: diabetes_model.joblib, diabetes_scaler.joblib")
    print("Now deploy these files along with your app to Render.")

if __name__ == "__main__":
    train_and_save_model()