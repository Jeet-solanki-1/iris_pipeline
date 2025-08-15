import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train(out_model="models/rf.joblib"):
    os.makedirs("models", exist_ok=True)
    train = pd.read_csv("data/processed/train.csv")
    X = train.drop(columns=["target"])
    y = train["target"]
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, out_model)
    print("Model saved to", out_model)

if __name__ == "__main__":
    train()
