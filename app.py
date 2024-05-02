# app.py

import warnings
import requests
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, request, jsonify, render_template
from models.calorie_model import train_model
from models.obesity_model import train_obesity_model
from models.dataset import calculate_bmi, map_activity_level
from datetime import datetime


app = Flask(__name__)

knn_calorie_model, calorie_label_encoder = train_model()
knn_obesity_model, obesity_label_encoder = train_obesity_model()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        age = int(request.form['age'])
        gender = request.form['gender']

        bmi = calculate_bmi(height, weight, age, gender)
        predicted_category = knn_obesity_model.predict([[bmi]])[0]
        predicted_calories = knn_calorie_model.predict([[age, weight, height/100, calorie_label_encoder.transform([gender])[0]]])[0]

        return render_template('result.html', predicted_calories=predicted_calories, bmi=predicted_category)
    
    return render_template('index.html')

@app.route('/predict_calories', methods=['POST'])
def predict_calories():
    data = request.get_json()
    height = float(data['height'])
    weight = float(data['weight'])
    age = int(data['age'])
    gender = data['gender']

    bmi = calculate_bmi(height, weight, age, gender)
    predicted_calories = knn_calorie_model.predict([[age, weight, height/100, calorie_label_encoder.transform([gender])[0]]])[0]

    response = {
        'predicted_calories': predicted_calories
    }
    return jsonify(response)

@app.route('/predict_obesity', methods=['POST'])
def predict_obesity():
    data = request.get_json()
    height = float(data['height'])
    weight = float(data['weight'])
    age = int(data['age'])
    gender = data['gender']
    activity_level = int(data['activity_level'])

    bmi = calculate_bmi(height, weight, age, gender)
    activity_category = map_activity_level(activity_level)
    predicted_category = knn_obesity_model.predict([[bmi]])[0]

    response = {
        'bmi': bmi,
        'activity_category': activity_category,
        'predicted_category': predicted_category
    }
    return jsonify(response)

@app.route('/predict', methods=['GET'])
def predict():
    # # Get user ID from your API
    # user_id_response = requests.get('http://localhost:8000/api/current-user-id')
    # user_id_data = user_id_response.json()
    # user_id = user_id_data['user_id']

    # Get user data from your API
    # user_data_response = requests.get(f'http://localhost:8000/user/{user_id}')
    user_data_response = requests.get('http://localhost:8000/user/1')
    user_data = user_data_response.json()
    # Extract relevant user information
    gender = user_data['gender']

    # Convert 'M' to 'male' and 'F' to 'female'
    if gender == 'male':
        gender = 'M'
    elif gender == 'female':
        gender = 'F'

    # Get health data from your API
    # health_data_response = requests.get(f'http://localhost:8000/health-data/{user_id}')
    health_data_response = requests.get('http://localhost:8000/health-data/1')
    health_data = health_data_response.json()
    # Extract relevant health data
    height = float(health_data['height'])
    weight = float(health_data['weight'])
    birthdate_str = health_data['birthdate']  # Get birthdate string

    birthdate = datetime.strptime(birthdate_str, '%Y-%m-%d')  # Convert birthdate string to datetime object
    today = datetime.now()  # Get current date
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))  # Calculate age

    bmi = calculate_bmi(height, weight, age, gender)
    predicted_category = knn_obesity_model.predict([[bmi]])[0]
    predicted_calories = knn_calorie_model.predict([[age, weight, height/100, calorie_label_encoder.transform([gender])[0]]])[0]

    return render_template('result.html', predicted_calories=predicted_calories, bmi=predicted_category)

if __name__ == "__main__":
    app.run(debug=True)