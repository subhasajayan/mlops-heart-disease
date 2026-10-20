# Preprocessing tests will go here
import pandas as pd
import joblib


preprocessor = joblib.load("models/preprocessor.pkl")


def test_preprocessor_loads():
    """Pipeline should load without errors."""
    assert preprocessor is not None


def test_preprocessor_transform_shape():
    """Pipeline should output a valid transformed feature matrix."""
    sample = pd.DataFrame({
        "age": [45],
        "sex": [1],
        "cp": [3],
        "trestbps": [120],
        "chol": [230],
        "fbs": [0],
        "restecg": [1],
        "thalach": [150],
        "exang": [0],
        "oldpeak": [2.3],
        "slope": [2],
        "ca": [0],
        "thal": [2]
    })

    transformed = preprocessor.transform(sample)
    assert transformed.shape[1] > 0
