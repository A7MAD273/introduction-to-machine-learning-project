"""Regression baseline: 3 models with default settings, evaluated by 3-fold CV on train only."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.data.split import SEED
from src.preprocessing.pipeline import build_preprocessor
from src.regression.dataset import get_regression_data

OUT_PATH = Path("outputs") / "regression_baseline.csv"


def baseline_models():
    """Default (untuned) configuration of each model."""
    return {
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(random_state=SEED, n_jobs=-1),
        "XGBoost": XGBRegressor(random_state=SEED, n_jobs=-1),
    }


def main():
    X_train, _, y_train, _ = get_regression_data()  # the test set is not touched here
    cv = KFold(n_splits=3, shuffle=True, random_state=SEED)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    rows = []
    for name, model in baseline_models().items():
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("model", model)])
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        rows.append({
            "model": name,
            "cv_mae": -res["test_mae"].mean(),
            "cv_rmse": -res["test_rmse"].mean(),
            "cv_r2": res["test_r2"].mean(),
            "fit_time_sec": res["fit_time"].mean(),
        })
        print(f"done: {name}")

    table = pd.DataFrame(rows).round(3)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_PATH, index=False)
    print(table.to_string(index=False))
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
