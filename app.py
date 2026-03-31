from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

lr_model = joblib.load('logistic_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[
        data['age'], data['gender'], data['chest_pain_type'],
        data['resting_bp'], data['serum_chol'], data['fasting_blood_sugar'],
        data['resting_ecg'], data['max_heart_rate'], data['exercise_angina'],
        data['oldpeak'], data['slope'], data['major_vessels']
    ]])
    lr_prob = float(lr_model.predict_proba(features)[0][1])
    rf_prob = float(rf_model.predict_proba(features)[0][1])
    avg_prob = (lr_prob + rf_prob) / 2
    if avg_prob >= 0.65:
        risk_level = "HIGH"
    elif avg_prob >= 0.35:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    return jsonify({
        'lr_probability': round(lr_prob, 2),
        'rf_probability': round(rf_prob, 2),
        'avg_probability': round(avg_prob, 2),
        'risk_level': risk_level,
        'risk_score': round(avg_prob * 10, 1)
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'VitalAI API running'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
