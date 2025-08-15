# iris\_pipeline — From raw flowers to a working ML pipeline

*Build, train, evaluate and visualize a simple ML pipeline on the Iris dataset — from first principles to reproducible outputs.*

---

Jeet, your AI lives.

This README gives you everything: why this project exists, how to set it up and run it, what each file does, the math and logic behind the pipeline, sample outputs, a flow diagram, and next steps to evolve the project into bigger AI work.

---

## TL;DR — What this repo does

* Loads the classic **Iris** dataset (built into `scikit-learn`).
* Saves raw data to `data/raw/iris.csv`.
* Processes and splits data into `data/processed/train.csv` (120 rows) and `data/processed/test.csv` (30 rows).
* Trains a baseline classifier (Random Forest) and saves it to `models/rf.joblib`.
* Evaluates on the held-out test set and prints metrics.
* Visualizes results with a confusion matrix.

**Goal:** teach the full ML cycle (data → features → model → eval → visualize) on a tiny reproducible dataset so you can scale these patterns to larger problems (NLP, CV, time series, etc.).

---

## Quick start (copy/paste — Windows CMD / PowerShell)

> Run these commands from the project root (where this README lives).

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Run the pipeline:

```bash
python src/data/make_dataset.py
python src/features/build_features.py
python src/models/train_model.py
python src/models/predict_model.py
python src/visualization/visualize.py
```

You should see `data/raw/iris.csv`, `data/processed/train.csv`, `data/processed/test.csv`, `models/rf.joblib`, a printed accuracy and classification report, and a confusion matrix window.

---

## Requirements

* Python 3.10 – 3.13 (this project was tested on Python 3.13. If you use 3.13 ensure wheels for your packages exist — they did in your environment.)
* `pip` for installing packages

**requirements.txt**

```
scikit-learn==1.7.1
pandas==2.3.1
matplotlib==3.10.5
joblib==1.5.1
numpy
```

(Exact versions are optional; use the ones that work on your machine.)

---

## Project structure

```
iris_pipeline/
├─ data/
│  ├─ raw/              # raw CSVs (iris.csv)
│  └─ processed/        # train.csv, test.csv
├─ models/              # saved model artifacts (rf.joblib)
├─ notebooks/           # optional: experiments / EDA
├─ src/
│  ├─ data/
│  │  └─ make_dataset.py
│  ├─ features/
│  │  └─ build_features.py
│  ├─ models/
│  │  ├─ train_model.py
│  │  └─ predict_model.py
│  └─ visualization/
│     └─ visualize.py
├─ requirements.txt
└─ README.md
```

---

## File-by-file: what each script does & how to call it

### `src/data/make_dataset.py`

* **Purpose:** Create `data/raw/iris.csv` from `sklearn.datasets.load_iris`.
* **Call:** `python src/data/make_dataset.py`
* **Why:** Keep a raw, reproducible snapshot of the dataset so you can always re-run the pipeline from the same origin.

### `src/features/build_features.py`

* **Purpose:** Read `data/raw/iris.csv`, optionally perform preprocessing, then split into train/test (80/20) and save to `data/processed/`.
* **Call:** `python src/features/build_features.py`
* **Why:** For robust evaluation, hold out a test set that the model never sees during training.

### `src/models/train_model.py`

* **Purpose:** Read `data/processed/train.csv`, train a RandomForestClassifier, and save the fitted model (`models/rf.joblib`).
* **Call:** `python src/models/train_model.py`
* **Why:** Build a baseline model you can compare more advanced methods against.

### `src/models/predict_model.py`

* **Purpose:** Load the saved model, run it on test data, print accuracy & classification report (precision, recall, f1).
* **Call:** `python src/models/predict_model.py`
* **Why:** Evaluate model generalization on unseen data.

### `src/visualization/visualize.py`

* **Purpose:** Plot a confusion matrix using `ConfusionMatrixDisplay.from_predictions`.
* **Call:** `python src/visualization/visualize.py`
* **Why:** Visual diagnostics quickly reveal which classes are getting confused.

---

## Libraries & key imports — who provides what

* `os` (Python stdlib): filesystem operations (`os.makedirs`).
* `pandas` (`pd`): tabular read/write and DataFrame manipulation (`pd.read_csv`, `df.to_csv`).
* `numpy` (`np`): numeric arrays and linear algebra operations.
* `scikit-learn`:

  * `sklearn.datasets.load_iris` — built-in dataset loader.
  * `sklearn.model_selection.train_test_split` — creates randomized train/test split.
  * `sklearn.ensemble.RandomForestClassifier` — the classifier used for training.
  * `sklearn.metrics` — `accuracy_score`, `classification_report`, `ConfusionMatrixDisplay`.
* `joblib`: efficient saving/loading of Python objects (models).

---

## Key logic & math (explained)

### 1) Train / Test split (why 120 / 30)

* We split the 150-row dataset using an **80/20** ratio:

  * Train size = `150 * 0.8 = 120`
  * Test size = `150 * 0.2 = 30`
* **Why split?** To judge generalization: a model should do well on unseen data, not just memorized examples.

**Implementation (sklearn):**

```python
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
```

* `stratify` preserves class proportions across splits.
* `random_state` fixes randomness for reproducibility.

---

### 2) Random Forest — core idea (intuitively)

* Random Forest is an *ensemble* of decision trees.
* Each tree is trained on a bootstrap sample (random rows) and considers random subsets of features at each split.
* Final prediction = majority vote across trees (for classification).
* Intuition: many weak, decorrelated learners combined produce a robust strong learner.

**Why used here:** Good off-the-shelf baseline; robust, fast, interpretable-ish (feature importances).

---

### 3) Evaluation metrics

* **Accuracy** = (correct predictions) / (total predictions).
* **Precision (per class)** = TP / (TP + FP).
  (When the model predicts class X, how often is it correct?)
* **Recall (per class)** = TP / (TP + FN).
  (Of all true X examples, how many did the model find?)
* **F1-score** = harmonic mean of precision & recall:
  `F1 = 2 * (precision * recall) / (precision + recall)`.
* **Confusion matrix**: table `M` where `M[i,j]` = count of samples with true class `i` predicted as `j`.

  * Diagonal = correct predictions.
  * Off-diagonal = types of errors (which classes get confused).

---

### 4) (Optional) Linear Discriminant Analysis (LDA) — Fisher’s idea

* LDA finds projection(s) that **maximize class separation** relative to class scatter.
* Objective: maximize `J(w) = (w^T S_B w) / (w^T S_W w)`
  where `S_B` is between-class scatter, `S_W` is within-class scatter.
* Solved via generalized eigenvalue problem: `S_W^{-1} S_B w = λ w`.
* We included an optional implementation earlier — useful for understanding dimensionality reduction & linear separability.

---

## What you should see when you run the pipeline

* `data/raw/iris.csv` (150 rows)
* `data/processed/train.csv` (120 rows)
* `data/processed/test.csv` (30 rows)
* `models/rf.joblib` — the trained model artifact.
* Terminal output:

  * Accuracy (e.g., `0.90`)
  * Classification report with precision/recall/f1 per class
* Confusion matrix plot — diagonal dominance indicates good performance.

---

## Flow diagram (simple ASCII)

```
[SKLEARN: load_iris] ---> data/raw/iris.csv
                              |
                              v
                    src/features/build_features.py
                              |
            (shuffle & split — stratify, random_state)
                              |
               +--------------+--------------+
               |                             |
      data/processed/train.csv      data/processed/test.csv
               |                             |
               v                             v
   src/models/train_model.py         src/models/predict_model.py
       (fit RandomForest)                  (load model, predict)
               |                             |
         models/rf.joblib               metrics output
                                             |
                                             v
                                     src/visualization/visualize.py
                                              (confusion matrix)
```

---

## What’s remaining / next steps (roadmap)

1. **Unit tests** (`tests/`) to assert file outputs, data shapes, model persists.
2. **Experiment logging** (CSV or MLflow) — record hyperparameters & metrics.
3. **Hyperparameter tuning** — `GridSearchCV`, `RandomizedSearchCV` or `Optuna`.
4. **Feature engineering** — scale features, add polynomial terms, inspect feature importances.
5. **Cross-validation** — k-fold CV for robust metric estimates.
6. **Model comparators** — SVM, Logistic Regression, XGBoost.
7. **Small neural net (Level 1)** — implement a tiny PyTorch MLP on same features.
8. **Packaging & API** — wrap model into a Flask or FastAPI service for demo UI.
9. **CI/CD** — GitHub Actions: run tests, lint, and auto-build artifacts.
10. **Dataset expansions** — move to real-world tasks: small NLP dataset, small CV dataset.
11. **Documentation & notebooks** — `notebooks/` with EDA and narrative write-up.

---

## Advanced: Deployment & production notes

* Save models with versioned filenames (e.g., `rf_v1.joblib`) and keep `experiments.csv`.
* For reproducible training, fix random seeds (`random_state`) and log package versions (`pip freeze` → `requirements.txt`).
* For production inference, prefer `joblib` or `pickle` only if you trust the environment — consider containerization.

---

## FAQs / Troubleshooting

* **Q:** `matplotlib` not found / plots don’t show?
  **A:** Install `matplotlib` (`python -m pip install matplotlib`) and ensure no display restrictions on headless servers (use `plt.savefig(...)` to save images).
* **Q:** `python` points to multiple installs?
  **A:** Use `where python` (Windows) and `python --version` to confirm. Use virtualenv to isolate environments.
* **Q:** Performance slow on 8GB laptop?
  **A:** Use smaller `n_estimators` (RandomForest param), limit `n_jobs`, and close other apps.

---

## Licensing & contribution

This repository is released under the **MIT License** (feel free to reuse and adapt — include attribution).

**Contributing:** open an issue or PR. Keep changes small, add tests for new functionality.

---

## Final notes — what you learned from the project

* The *full ML lifecycle* (data ingestion → preprocessing → train → evaluate → visualize).
* Why we **hold out** a test set and how to compute/interpret classification metrics.
* How a baseline model works and how to persist it for later reuse.
* How to visualize errors with a confusion matrix and read what the matrix tells you.

