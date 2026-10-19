import joblib
import pandas as pd


def load_artifacts():
    model = joblib.load("models/model.pkl")
    return model


def predict(input_dict):
    model = load_artifacts()

    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)[0]

    return int(prediction)


if __name__ == "__main__":
    sample = {
        "age": 45,
        "sex": 1,
        "cp": 3,
        "trestbps": 120,
        "chol": 230,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 2,
        "ca": 0,
        "thal": 2
    }

    result = predict(sample)
    print("Prediction:", result)
