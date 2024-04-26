# calorie_model.py

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from .dataset import load_data

def train_model():
    data = load_data("/datasets/Dataset.csv")
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['gender'])
    X = data[['age', 'weight(kg)', 'height(m)', 'Gender']]
    y = data['calories_to_maintain_weight']
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X, y)
    return knn_model, label_encoder