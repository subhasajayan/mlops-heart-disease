# scripts/download_data.py

import os
import requests

def main():
    # URL for the Statlog Heart dataset (text file)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"

    # Create data/raw folder if it doesn't exist
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)


    # Target file path
    target_path = os.path.join(raw_dir, "heart.dat")

    print(f"Downloading dataset from: {url}")
    response = requests.get(url)
    response.raise_for_status()  # will raise error if download fails

    with open(target_path, "wb") as f:
        f.write(response.content)

    print(f"Saved dataset to: {target_path}")

if __name__ == "__main__":
    main()
