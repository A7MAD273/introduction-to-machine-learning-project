"""Classification: predict IsCanceled (will the booking be canceled?)."""

from src.data.split import SEED, split_data
from src.features.selection import CLASSIFICATION_FEATURES, CLASSIFICATION_TARGET
from src.preprocessing.pipeline import build_preprocessor, load_prepared_data

df = load_prepared_data()
train, test = split_data(df, stratify_col="IsCanceled")

X_train, y_train = train[CLASSIFICATION_FEATURES], train[CLASSIFICATION_TARGET]
X_test, y_test = test[CLASSIFICATION_FEATURES], test[CLASSIFICATION_TARGET]