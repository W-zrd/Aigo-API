# models/obesity_model.py

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from .dataset import load_obesity_data

def train_obesity_model():
    obesity_data = load_obesity_data("datasets/obesity_data.csv")
    label_encoder = LabelEncoder()
    obesity_data['Gender'] = label_encoder.fit_transform(obesity_data['Gender'])
    X = obesity_data[['BMI']]
    y = obesity_data['ObesityCategory']
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X, y)
    return knn_model, label_encoder