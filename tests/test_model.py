import joblib
import pandas as pd


model = joblib.load("models/model.pkl")


def test_model_loads():
    assert model is not None


def test_model_prediction_shape():
    sample = pd.DataFrame(
        {
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
            "thal": [2],
        }
    )

    pred = model.predict(sample)
    assert pred.shape == (1,)


def test_model_prediction_values():
    sample = pd.DataFrame(
        {
            "age": [60],
            "sex": [0],
            "cp": [2],
            "trestbps": [140],
            "chol": [260],
            "fbs": [1],
            "restecg": [0],
            "thalach": [120],
            "exang": [1],
            "oldpeak": [1.5],
            "slope": [1],
            "ca": [2],
            "thal": [3],
        }
    )

    pred = model.predict(sample)
    assert pred[0] in [0, 1]
