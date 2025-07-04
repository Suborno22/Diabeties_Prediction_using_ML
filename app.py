import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Import prediction function only when needed
def get_predictor():
    """Lazy import to avoid loading ML libraries at startup."""
    from diabetes_predictor import predict_diabetes
    return predict_diabetes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
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
        
        # Import and use predictor only when needed
        predict_diabetes = get_predictor()
        result = predict_diabetes(data)
        
        return jsonify({'result': result})
    
    except Exception as e:
        return jsonify({'error': str(e), 'result': 'Error in prediction'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)