# app.py

import warnings
import pickle
from flask import Flask, request, jsonify, render_template
from models.calorie_model import train_model
from models.obesity_model import train_obesity_model
from models.dataset import calculate_bmi, map_activity_level
from models.Execise import load_exercise_data, kg_to_lb, rekomendasi_aktivitas
from datetime import datetime

app = Flask(__name__)

knn_calorie_model, calorie_label_encoder = train_model()
knn_obesity_model, obesity_label_encoder = train_obesity_model()

# Load exercise data
exercise_data = load_exercise_data("datasets/exercise_dataset.csv")


# Route for predicting calorie
@app.route('/api/predict/calorie', methods=['POST'])
def predict_calorie():
    data = request.json
    height = float(data['height'])
    weight = float(data['weight'])
    age = int(data['age'])
    gender = data['gender']

    bmi = calculate_bmi(height, weight, age, gender)
    # predicted_calories = knn_calorie_model.predict([[age, weight, height/100, calorie_label_encoder.transform([gender])[0]]])[0]
    with open('datasets/calories_classifier.pkl', 'rb') as file:
        clf = pickle.load(file)

    predicted_pkl = clf.predict([[age, weight, height/100, calorie_label_encoder.transform([gender])[0]]])[0]

    return jsonify({'predicted_calories': predicted_pkl})
    # return jsonify({'predicted_calories': predicted_calories})

# Route for predicting obesity
@app.route('/api/predict/obesity', methods=['POST'])
def predict_obesity():
    data = request.json
    height = float(data['height'])
    weight = float(data['weight'])
    age = int(data['age'])
    gender = data['gender']
    activity_level = int(data['activity_level'])

    bmi = calculate_bmi(height, weight, age, gender)
    activity_category = map_activity_level(activity_level)
    # predicted_category = knn_obesity_model.predict([[bmi]])[0]
    with open('datasets/obesity_classifier.pkl', 'rb') as file:
        clf = pickle.load(file)
    
    predicted_pkl = clf.predict([[bmi]])[0]

    return jsonify({'predicted_category': predicted_pkl, 'bmi': bmi, 'activity_category': activity_category})

# Route for predicting exercise
@app.route('/api/predict/exercise', methods=['POST'])
def predict_exerices():
    data = request.json
    weight = float(data['weight'])
    calorie = float(data['calorie'])

    recommendations = rekomendasi_aktivitas(exercise_data, calorie, weight)
    
    return jsonify(recommendations)

    

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
