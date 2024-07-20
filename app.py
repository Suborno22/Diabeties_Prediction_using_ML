from flask import Flask, render_template, request, jsonify
from diabetes_predictor import predict_diabetes

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [
        float(request.form['pregnancies']),
        float(request.form['glucose']),
        float(request.form['blood_pressure']),
        float(request.form['skin_thickness']),
        float(request.form['insulin']),
        float(request.form['bmi']),
        float(request.form['diabetes_pedigree_function']),
        float(request.form['age'])
    ]
    result = predict_diabetes(data)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)