"""Classification: compare original features, PCA features and LDA features.

Uses the tuned settings saved by src.classification.models
(outputs/tables/classification_tuned_cv.csv), so run that first.
PCA and LDA are steps inside the Pipeline, so in CV they are fitted
on the training folds only (LDA uses the labels, so this matters).
"""

import ast
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from src.classification.models import (
    BASELINE_MODELS,
    CV,
    SCORING,
    TABLES_DIR,
    X_train,
    y_train,
)
from src.data.split import SEED
from src.preprocessing.pipeline import build_preprocessor

FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"

# Keep enough principal components to explain 95% of the variance.
PCA_VARIANCE = 0.95


def tuned_models():
    """The three models with the best settings found by GridSearchCV."""
    tuned = pd.read_csv(TABLES_DIR / "classification_tuned_cv.csv")
    models = {}
    for _, row in tuned.iterrows():
        params = ast.literal_eval(row["best_params"])
        models[row["model"]] = clone(BASELINE_MODELS[row["model"]]).set_params(**params)
    return models


def make_pipeline(model, reduction):
    """Preprocessing -> (PCA or LDA or nothing) -> model."""
    steps = [("prep", build_preprocessor(X_train))]
    if reduction == "pca":
        steps.append(("reduce", PCA(n_components=PCA_VARIANCE, random_state=SEED)))
    elif reduction == "lda":
        # 2 classes -> LDA gives at most 1 component.
        steps.append(("reduce", LinearDiscriminantAnalysis(n_components=1)))
    steps.append(("model", model))
    return Pipeline(steps)


def pca_explained_variance():
    """Fit PCA on the preprocessed train set. Return a table and save a plot."""
    X = build_preprocessor(X_train).fit_transform(X_train)
    pca = PCA(random_state=SEED).fit(X)
    ratio = pca.explained_variance_ratio_
    table = pd.DataFrame({
        "component": np.arange(1, len(ratio) + 1),
        "explained_variance": ratio,
        "cumulative": ratio.cumsum(),
    })
    n_keep = int(np.searchsorted(ratio.cumsum(), PCA_VARIANCE) + 1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(table["component"], table["explained_variance"], color="#4C72B0", label="each component")
    ax.plot(table["component"], table["cumulative"], color="#C44E52", marker=".", label="cumulative")
    ax.axhline(PCA_VARIANCE, color="gray", ls="--", label=f"{PCA_VARIANCE:.0%}")
    ax.axvline(n_keep, color="green", ls="--", label=f"{n_keep} components kept")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(f"PCA on classification features ({X.shape[1]} columns)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "classification_pca_variance.png", dpi=150)
    plt.close(fig)
    return table, n_keep, X.shape[1]


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    variance, n_keep, n_columns = pca_explained_variance()
    variance.round(4).to_csv(TABLES_DIR / "classification_pca_variance.csv", index=False)
    print(f"PCA: {n_columns} columns -> {n_keep} components explain {PCA_VARIANCE:.0%} of the variance")
    print("Saved -> outputs/figures/classification_pca_variance.png")

    rows = []
    for name, model in tuned_models().items():
        for reduction in ["original", "pca", "lda"]:
            print("Training", name, "with", reduction, "features ...")
            start = time.time()
            pipeline = make_pipeline(clone(model), reduction)
            scores = cross_validate(pipeline, X_train, y_train, cv=CV, scoring=SCORING)
            row = {m: scores[f"test_{m}"].mean() for m in SCORING}
            row["n_features"] = {"original": n_columns, "pca": n_keep, "lda": 1}[reduction]
            row["time_sec"] = time.time() - start
            rows.append({"model": name, "features": reduction, **row})

    table = pd.DataFrame(rows).round(4)
    table.to_csv(TABLES_DIR / "classification_pca_lda_cv.csv", index=False)
    print(table.to_string(index=False))
    print("Saved -> outputs/tables/classification_pca_lda_cv.csv")
