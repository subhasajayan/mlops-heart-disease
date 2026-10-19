import pandas as pd
from src.train import preprocess, train_model


def test_preprocess():
    df = pd.DataFrame(
        {
            "age": [50],
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
            "target": [1],
        }
    )

    X, y, scaler = preprocess(df)
    assert X.shape[0] == 1
    assert y.iloc[0] == 1
    assert scaler is not None


def test_train_model():
    df = pd.DataFrame(
        {
            "age": [50, 60],
            "sex": [1, 0],
            "cp": [3, 2],
            "trestbps": [120, 130],
            "chol": [230, 250],
            "fbs": [0, 1],
            "restecg": [1, 0],
            "thalach": [150, 140],
            "exang": [0, 1],
            "oldpeak": [2.3, 1.5],
            "slope": [2, 1],
            "ca": [0, 2],
            "thal": [2, 3],
            "target": [1, 0],
        }
    )

    X, y, scaler = preprocess(df)
    model = train_model(X, y)
    assert model is not None
