import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import joblib

def visualize(model_path="models/rf.joblib", test_csv="data/processed/test.csv"):
    clf = joblib.load(model_path)
    test = pd.read_csv(test_csv)
    X_test = test.drop(columns=["target"])
    y_true = test["target"]
    y_pred = clf.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    visualize()
