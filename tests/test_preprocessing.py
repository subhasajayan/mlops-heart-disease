import pandas as pd
from src.preprocess import clean_data


def test_clean_data_replaces_missing():
    df = pd.DataFrame(
        {
            "age": ["?", 50],
            "sex": [1, "?"],
            "cp": [3, 2],
            "trestbps": [120, 130],
            "chol": [230, "?"],
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

    cleaned = clean_data(df)
    assert cleaned.isna().sum().sum() == 0


def test_clean_data_numeric_conversion():
    df = pd.DataFrame(
        {
            "age": ["50"],
            "sex": ["1"],
            "cp": ["3"],
            "trestbps": ["120"],
            "chol": ["230"],
            "fbs": ["0"],
            "restecg": ["1"],
            "thalach": ["150"],
            "exang": ["0"],
            "oldpeak": ["2.3"],
            "slope": ["2"],
            "ca": ["0"],
            "thal": ["2"],
            "target": ["1"],
        }
    )

    cleaned = clean_data(df)
    assert cleaned.dtypes["age"] == "int64"
