"""Classification: predict IsCanceled (will the booking be canceled?)."""

import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.split import SEED, split_data
from src.features.selection import CLASSIFICATION_FEATURES, CLASSIFICATION_TARGET
from src.preprocessing.pipeline import build_preprocessor, load_prepared_data

df = load_prepared_data()
train, test = split_data(df, stratify_col="IsCanceled")

X_train, y_train = train[CLASSIFICATION_FEATURES], train[CLASSIFICATION_TARGET]
X_test, y_test = test[CLASSIFICATION_FEATURES], test[CLASSIFICATION_TARGET]

TABLES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "tables"

# Metrics required by the spec. "average_precision" is PR-AUC.
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}

# Same 5 folds for every model, so the comparison is fair.
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Baseline = default settings (only the seed is fixed).
BASELINE_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "random_forest": RandomForestClassifier(random_state=SEED, n_jobs=-1),
    "xgboost": XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss"),
}


def make_pipeline(model):
    """Preprocessing + model. Inside CV the preprocessing is fitted on the training folds only."""
    return Pipeline([("prep", build_preprocessor(X_train)), ("model", model)])


def cv_scores(model):
    """5-fold CV on the train set. Return the mean of each metric and the time."""
    start = time.time()
    scores = cross_validate(make_pipeline(model), X_train, y_train, cv=CV, scoring=SCORING)
    row = {name: scores[f"test_{name}"].mean() for name in SCORING}
    row["time_sec"] = time.time() - start
    return row


if __name__ == "__main__":
    rows = []
    for name, model in BASELINE_MODELS.items():
        print("Training", name, "...")
        rows.append({"model": name, "config": "baseline", **cv_scores(model)})

    table = pd.DataFrame(rows).round(4)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "classification_baseline_cv.csv", index=False)

    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    print("Saved -> outputs/tables/classification_baseline_cv.csv")
