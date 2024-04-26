# app.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, request, jsonify, render_template
from models.calorie_model import train_model
from models.obesity_model import train_obesity_model
from models.dataset import calculate_bmi, map_activity_level

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
        predicted_calories = knn_calorie_model.predict([[age, weight, height/100, calorie_label_encoder.transform([gender])[0]]])[0]

        return render_template('result.html', predicted_calories=predicted_calories)
    
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

if __name__ == "__main__":
    app.run(debug=True)