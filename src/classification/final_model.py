"""Classification: choose the deployed model, test it once, and save it.

Selection rule: the tuned model with the highest mean CV F1
(outputs/tables/classification_tuned_cv.csv), using the original features
(they beat PCA and LDA in outputs/tables/classification_pca_lda_cv.csv).
The test set is used here only, once, after everything was chosen on CV.

Run src.classification.models first.
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.classification.models import TABLES_DIR, X_test, X_train, y_test, y_train
from src.classification.pca_lda import make_pipeline, tuned_models
from src.features.selection import CLASSIFICATION_FEATURES

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "outputs" / "figures"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "classification_model.joblib"
INFO_PATH = ARTIFACTS_DIR / "classification_model_info.json"

SELECTION_METRIC = "f1"


def choose_model():
    """Return the name of the tuned model with the best CV F1."""
    tuned = pd.read_csv(TABLES_DIR / "classification_tuned_cv.csv")
    best = tuned.sort_values(SELECTION_METRIC, ascending=False).iloc[0]
    return best["model"], tuned


def test_scores(y_true, y_pred, y_prob):
    """All required metrics on the test set."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def save_plots(name, y_true, y_pred, y_prob):
    """Confusion matrix, ROC curve and precision-recall curve on the test set."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Not canceled", "Canceled"], cmap="Blues", ax=axes[0]
    )
    axes[0].set_title("Confusion matrix")
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], color="gray", ls="--")
    axes[1].set_title("ROC curve")
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=axes[2])
    axes[2].set_title("Precision-recall curve")
    fig.suptitle(f"Final classification model on the test set: {name}")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "classification_test_results.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    name, tuned = choose_model()
    print("Selected model:", name, f"(highest CV {SELECTION_METRIC})")

    # Fit on the whole train set with the tuned settings.
    pipeline = make_pipeline(tuned_models()[name], "original")
    pipeline.fit(X_train, y_train)

    # The only time the test set is used.
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    scores = test_scores(y_test, y_pred, y_prob)
    cv_row = tuned[tuned["model"] == name].iloc[0]

    table = pd.DataFrame([
        {"model": name, "set": "cv (train)", **{m: cv_row[m] for m in scores}},
        {"model": name, "set": "test", **scores},
    ]).round(4)
    table.to_csv(TABLES_DIR / "classification_test_results.csv", index=False)
    print(table.to_string(index=False))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(confusion_matrix(y_test, y_pred))
    print("Saved -> outputs/tables/classification_test_results.csv")

    save_plots(name, y_test, y_pred, y_prob)
    print("Saved -> outputs/figures/classification_test_results.png")

    # Save the whole pipeline (preprocessing + model) for inference.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    info = {
        "model": name,
        "selection": f"highest mean 5-fold CV {SELECTION_METRIC} among tuned models",
        "params": cv_row["best_params"],
        "features": CLASSIFICATION_FEATURES,
        "test_scores": {m: round(v, 4) for m, v in scores.items()},
    }
    INFO_PATH.write_text(json.dumps(info, indent=2))
    print("Saved -> artifacts/classification_model.joblib")
    print("Saved -> artifacts/classification_model_info.json")
