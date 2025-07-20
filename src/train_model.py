import pandas as pd
from xgboost import XGBClassifier
import joblib
import shap

def load_data():
    # Dummy data loading function
    from sklearn.datasets import fetch_20newsgroups_vectorized
    data = fetch_20newsgroups_vectorized(subset='train')
    return data.data, data.target

def train():
    X_train, y_train = load_data()
    model = XGBClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, 'model.joblib')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    joblib.dump(shap_values, 'shap_values.pkl')

if __name__ == "__main__":
    train()
