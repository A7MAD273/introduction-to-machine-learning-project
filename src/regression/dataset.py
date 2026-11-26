"""Regression data: predict ADR (price per night) of paid bookings."""

from src.data.split import split_data
from src.features.selection import REGRESSION_FEATURES, REGRESSION_TARGET
from src.preprocessing.pipeline import load_prepared_data


def get_regression_data():
    """Return X_train, X_test, y_train, y_test for the ADR task."""
    df = load_prepared_data()
    train, test = split_data(df, stratify_col="IsCanceled")

    # Scope of the task: paid bookings only (ADR == 0 are free/complimentary stays).
    # This is a fixed rule, not learned from data, so it is applied to both sets.
    train = train[train[REGRESSION_TARGET] > 0].reset_index(drop=True)
    test = test[test[REGRESSION_TARGET] > 0].reset_index(drop=True)

    X_train, y_train = train[REGRESSION_FEATURES], train[REGRESSION_TARGET]
    X_test, y_test = test[REGRESSION_FEATURES], test[REGRESSION_TARGET]
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_regression_data()
    print("X_train:", X_train.shape, "| X_test:", X_test.shape)
    print("ADR min (train):", y_train.min(), "| ADR min (test):", y_test.min())
    assert (y_train > 0).all() and (y_test > 0).all()
    print("All checks passed.")
