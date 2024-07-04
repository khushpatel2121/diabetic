from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from joblib import load
import os

# Load the trained model
model = load('diabetes_model.joblib')

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # Handle preflight request for CORS
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS"
        }
        return Response(headers=headers)

    data = request.get_json(force=True)
    features = [data['Pregnancies'], data['Glucose'], data['BloodPressure'], data['SkinThickness'], data['Insulin'], data['BMI'], data['DiabetesPedigreeFunction'], data['Age']]
    prediction = model.predict([features])
    result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'

    response = jsonify({'prediction': result})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")

    return response

# Use environment variables to configure host and port for deployment flexibility
if __name__ == '__main__':
    # Use PORT environment variable if defined, otherwise default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
