import os
import pandas as pd
from sklearn.datasets import load_iris

def make_iris_dataset(out_dir="data/raw"):
    os.makedirs(out_dir, exist_ok=True)
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.to_csv(os.path.join(out_dir, "iris.csv"), index=False)
    print("Saved iris.csv to", out_dir)

if __name__ == "__main__":
    make_iris_dataset()
