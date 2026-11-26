"""Regression tuning: RandomizedSearchCV for Ridge, RandomForest, XGBoost (train only)."""

import json
import time
from pathlib import Path

import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.data.split import SEED
from src.preprocessing.pipeline import build_preprocessor
from src.regression.dataset import get_regression_data

N_ITER = 20          # random combinations tried per model
CV_FOLDS = 3
OPTIMIZE_METRIC = "mae"   # the metric used to pick the best configuration
OUT_CSV = Path("outputs") / "regression_tuning.csv"
OUT_JSON = Path("outputs") / "regression_best_params.json"

SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}

# Search spaces. "model__" means: a parameter of the model step inside the Pipeline.
SEARCH_SPACES = {
    "Ridge": (
        Ridge(),
        {"model__alpha": loguniform(1e-2, 1e2)},
    ),
    "RandomForest": (
        RandomForestRegressor(random_state=SEED, n_jobs=-1),
        {
            "model__n_estimators": randint(100, 400),
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_split": randint(2, 11),
            "model__min_samples_leaf": randint(1, 6),
            "model__max_features": [1.0, 0.5, 0.3],
        },
    ),
    "XGBoost": (
        XGBRegressor(random_state=SEED, n_jobs=-1),
        {
            "model__n_estimators": randint(200, 700),
            "model__max_depth": randint(3, 11),
            "model__learning_rate": loguniform(0.02, 0.3),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.5, 0.5),
            "model__min_child_weight": randint(1, 11),
        },
    ),
}


def clean_params(params):
    """Convert numpy numbers to plain Python so they can be saved as JSON."""
    out = {}
    for key, value in params.items():
        value = value.item() if hasattr(value, "item") else value
        out[key.replace("model__", "")] = value
    return out


def main():
    X_train, _, y_train, _ = get_regression_data()   # the test set is not touched here
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    baseline = pd.read_csv(Path("outputs") / "regression_baseline.csv").set_index("model")

    rows, best_params = [], {}
    for name, (model, space) in SEARCH_SPACES.items():
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("model", model)])
        search = RandomizedSearchCV(
            pipe, space, n_iter=N_ITER, scoring=SCORING, refit=OPTIMIZE_METRIC,
            cv=cv, random_state=SEED, n_jobs=1, verbose=1,
        )
        start = time.time()
        search.fit(X_train, y_train)
        search_time = time.time() - start

        i = search.best_index_
        res = search.cv_results_
        rows.append({
            "model": name,
            "baseline_cv_mae": baseline.loc[name, "cv_mae"],
            "tuned_cv_mae": -res["mean_test_mae"][i],
            "tuned_cv_rmse": -res["mean_test_rmse"][i],
            "tuned_cv_r2": res["mean_test_r2"][i],
            "tuned_fit_time_sec": res["mean_fit_time"][i],
            "search_time_sec": search_time,
        })
        best_params[name] = clean_params(search.best_params_)
        print(f"done: {name} | best CV MAE = {-res['mean_test_mae'][i]:.3f}")

    table = pd.DataFrame(rows).round(3)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print(table.to_string(index=False))
    print("saved:", OUT_CSV, "and", OUT_JSON)


if __name__ == "__main__":
    main()
