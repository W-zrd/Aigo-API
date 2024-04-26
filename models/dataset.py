# dataset.py

import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def load_obesity_data(file_path):
    df = pd.read_csv(file_path)
    return df

def calculate_bmi(height, weight, age, gender):
    bmi = weight / ((height/100) ** 2)
    
    if gender.lower() == 'male':
        bmi += (0.03 * age)
    elif gender.lower() == 'female':
        bmi += (0.02 * age)
    
    return bmi

def map_activity_level(activity_level):
    activity_mapping = {1: 'Low', 2: 'Moderate', 3: 'High', 4: 'Very High'}
    return activity_mapping.get(activity_level, 'Unknown')