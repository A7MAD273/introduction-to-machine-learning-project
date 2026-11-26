"""Classification: predict IsCanceled (will the booking be canceled?)."""

import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
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

# Search space for each model. "model__" points to the model step of the Pipeline.
# class_weight / scale_pos_weight give more weight to the canceled class (30%),
# which raises recall. scale_pos_weight 2.3 = not canceled / canceled (70 / 30).
PARAM_GRIDS = {
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1, 10],
        "model__class_weight": [None, "balanced"],
    },
    "random_forest": {
        "model__n_estimators": [300],
        "model__max_depth": [None, 20],
        "model__min_samples_leaf": [1, 3, 5],
        "model__max_features": ["sqrt", 0.5],
        "model__class_weight": [None, "balanced"],
    },
    "xgboost": {
        "model__n_estimators": [300, 600],
        "model__max_depth": [4, 6, 8],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__scale_pos_weight": [1, 2.3],
    },
}

# The best configuration is chosen by F1 (classes are not balanced, so accuracy is misleading).
TUNING_METRIC = "f1"


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


def tune(name):
    """GridSearchCV on the train set (same 5 folds). Return the fitted search."""
    search = GridSearchCV(
        make_pipeline(BASELINE_MODELS[name]),
        PARAM_GRIDS[name],
        scoring=SCORING,
        refit=TUNING_METRIC,
        cv=CV,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def tuned_scores(search):
    """CV metrics of the best configuration, taken from the search results."""
    best = search.best_index_
    results = search.cv_results_
    row = {name: results[f"mean_test_{name}"][best] for name in SCORING}
    params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    row["best_params"] = str(params)
    return row


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: baselines
    rows = []
    for name, model in BASELINE_MODELS.items():
        print("Training", name, "...")
        rows.append({"model": name, "config": "baseline", **cv_scores(model)})

    table = pd.DataFrame(rows).round(4)
    table.to_csv(TABLES_DIR / "classification_baseline_cv.csv", index=False)
    print(table.to_string(index=False))
    print("Saved -> outputs/tables/classification_baseline_cv.csv")

    # Step 2: tuning
    rows = []
    for name in BASELINE_MODELS:
        print("Tuning", name, "...")
        start = time.time()
        search = tune(name)
        row = tuned_scores(search)
        row["time_sec"] = time.time() - start
        rows.append({"model": name, "config": "tuned", **row})

    table = pd.DataFrame(rows).round(4)
    table.to_csv(TABLES_DIR / "classification_tuned_cv.csv", index=False)
    print(table.to_string(index=False))
    print("Saved -> outputs/tables/classification_tuned_cv.csv")
