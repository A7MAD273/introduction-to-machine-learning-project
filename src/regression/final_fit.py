"""Final fit: train each tuned model on the full train set and save the pipelines."""

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.data.split import SEED
from src.preprocessing.pipeline import build_preprocessor
from src.regression.dataset import get_regression_data

ARTIFACTS = Path("artifacts")
OUTPUTS = Path("outputs")
BEST_PARAMS = OUTPUTS / "regression_best_params.json"
TUNING_CSV = OUTPUTS / "regression_tuning.csv"
SELECTED_JSON = OUTPUTS / "regression_selected.json"
FINAL_FIT_CSV = OUTPUTS / "regression_final_fit.csv"

MODEL_CLASSES = {
    "Ridge": lambda: Ridge(),
    "RandomForest": lambda: RandomForestRegressor(random_state=SEED, n_jobs=-1),
    "XGBoost": lambda: XGBRegressor(random_state=SEED, n_jobs=-1),
}
FILE_NAMES = {
    "Ridge": "regression_ridge.joblib",
    "RandomForest": "regression_random_forest.joblib",
    "XGBoost": "regression_xgboost.joblib",
}


def main():
    X_train, _, y_train, _ = get_regression_data()   # the test set is not touched here
    best_params = json.loads(BEST_PARAMS.read_text(encoding="utf-8"))
    ARTIFACTS.mkdir(exist_ok=True)

    rows = []
    for name, make_model in MODEL_CLASSES.items():
        model = make_model().set_params(**best_params[name])
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("model", model)])

        start = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - start

        path = ARTIFACTS / FILE_NAMES[name]
        joblib.dump(pipe, path, compress=3)
        size_mb = path.stat().st_size / 1e6
        rows.append({"model": name, "train_time_sec": train_time,
                     "file": str(path), "size_mb": size_mb})
        print(f"saved {path} ({size_mb:.1f} MB, fit {train_time:.1f}s)")

    pd.DataFrame(rows).round(3).to_csv(FINAL_FIT_CSV, index=False)

    # Selection criterion: lowest tuned cross-validation MAE (decided on train data only)
    tuning = pd.read_csv(TUNING_CSV)
    winner = tuning.loc[tuning["tuned_cv_mae"].idxmin()]
    selected = {
        "selected_model": winner["model"],
        "criterion": "lowest tuned 3-fold CV MAE on the training set",
        "tuned_cv_mae": float(winner["tuned_cv_mae"]),
        "artifact": str(ARTIFACTS / FILE_NAMES[winner["model"]]),
        "best_params": best_params[winner["model"]],
    }
    SELECTED_JSON.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print("selected model:", selected["selected_model"])


if __name__ == "__main__":
    main()
