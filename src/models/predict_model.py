import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def predict(model_path="models/rf.joblib", test_csv="data/processed/test.csv"):
    clf = joblib.load(model_path)
    test = pd.read_csv(test_csv)
    X_test = test.drop(columns=["target"])
    y_true = test["target"]
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    predict()
