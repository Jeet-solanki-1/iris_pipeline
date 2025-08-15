import os
import pandas as pd
from sklearn.model_selection import train_test_split

def build_features(raw_csv="data/raw/iris.csv", out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(raw_csv)
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print("Processed data saved to", out_dir)

if __name__ == "__main__":
    build_features()
