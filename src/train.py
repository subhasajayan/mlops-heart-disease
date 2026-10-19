import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

def load_data():
    df = pd.read_csv("data/heart.csv")   # adjust path if needed
    return df

def preprocess(df):
    X = df.drop("target", axis=1)        # adjust column name if needed
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def train_model(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def save_artifacts(model, scaler):
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "preprocessor.pkl")

if __name__ == "__main__":
    df = load_data()
    X, y, scaler = preprocess(df)
    model = train_model(X, y)
    save_artifacts(model, scaler)
