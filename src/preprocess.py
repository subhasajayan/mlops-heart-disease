import pandas as pd
import os

def load_raw_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "data", "raw", "heart.dat")

    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal", "target"
    ]

    df = pd.read_csv(raw_path, sep=" ", names=columns)
    return df

def clean_data(df):
    # Replace '?' with NaN if present
    df = df.replace("?", pd.NA)

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Handle missing values (simple strategy)
    df = df.fillna(df.median())

    return df

def save_clean_data(df):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    df.to_csv(os.path.join(processed_dir, "heart_clean.csv"), index=False)

def main():
    df = load_raw_data()
    df = clean_data(df)
    save_clean_data(df)
    print("Cleaned dataset saved to data/processed/heart_clean.csv")

if __name__ == "__main__":
    main()
