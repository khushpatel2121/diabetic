from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load

# Load the trained model
model = load('/Users/khushpatel/Desktop/IBM/Api/diabetes_model.joblib')

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [data['Pregnancies'], data['Glucose'], data['BloodPressure'], data['SkinThickness'], data['Insulin'], data['BMI'], data['DiabetesPedigreeFunction'], data['Age']]
    prediction = model.predict([features])
    result = 'Diabatic' if prediction[0] == 1 else 'Non-Diabatic'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)