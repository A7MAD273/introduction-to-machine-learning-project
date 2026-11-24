"""Regression: predict ADR (price per night)."""

from src.data.split import SEED, split_data
from src.features.selection import REGRESSION_FEATURES, REGRESSION_TARGET
from src.preprocessing.pipeline import build_preprocessor, load_prepared_data

df = load_prepared_data()
train, test = split_data(df, stratify_col="IsCanceled")

X_train, y_train = train[REGRESSION_FEATURES], train[REGRESSION_TARGET]
X_test, y_test = test[REGRESSION_FEATURES], test[REGRESSION_TARGET]