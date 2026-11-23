"""Preprocessing shared by all three tasks.

Everything that learns from the data (fill values, rare categories, scaling)
is inside the transformer returned by build_preprocessor(). Put it inside each
model's Pipeline, so it is fitted on the training data only.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_data import load_raw_data
from src.features.selection import add_features
from src.preprocessing.clean import clean_data

# Categories seen in less than 1% of the training rows are grouped together
# (e.g. the many small agents become one "infrequent" column).
MIN_CATEGORY_FREQUENCY = 0.01


def load_prepared_data():
    """Load H2.csv, clean it, and add engineered features (HasCompany)."""
    return add_features(clean_data(load_raw_data()))


def build_preprocessor(X):
    """Return an unfitted transformer for the columns of X.

    Numbers: fill missing with the median, then scale.
    Text: fill missing with "None", then one-hot encode, grouping rare categories.
    """
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="None")),
        ("encode", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=MIN_CATEGORY_FREQUENCY,
            sparse_output=False,
        )),
    ])
    return ColumnTransformer([
        ("num", numeric_steps, numeric),
        ("cat", categorical_steps, categorical),
    ])


if __name__ == "__main__":
    import numpy as np

    from src.data.split import split_data
    from src.features.selection import (
        CLASSIFICATION_FEATURES,
        CLUSTERING_FEATURES,
        REGRESSION_FEATURES,
    )

    df = load_prepared_data()
    train, test = split_data(df, stratify_col="IsCanceled")

    tasks = {
        "Regression": REGRESSION_FEATURES,
        "Classification": CLASSIFICATION_FEATURES,
        "Clustering": CLUSTERING_FEATURES,
    }
    for name, features in tasks.items():
        prep = build_preprocessor(train[features])
        X_train = prep.fit_transform(train[features])  # learns from train only
        X_test = prep.transform(test[features])
        assert not np.isnan(X_train).any() and not np.isnan(X_test).any()
        print(f"{name}: {len(features)} features -> {X_train.shape[1]} columns "
              f"(train {X_train.shape[0]}, test {X_test.shape[0]})")

    assert X_train.shape[1] <= len(df) // 100  # clustering: >= 100 rows per variable
    print("No missing values after preprocessing. All checks passed.")