from sklearn.model_selection import train_test_split

from src.data.load_data import load_raw_data
from src.preprocessing.clean import clean_data

SEED = 42
TEST_SIZE = 0.2


def split_data(df, stratify_col=None):
    """Split the cleaned data into train (80%) and test (20%)."""
    stratify = df[stratify_col] if stratify_col else None
    train, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=stratify
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    df = clean_data(load_raw_data())
    train, test = split_data(df, stratify_col="IsCanceled")
    print("Train:", train.shape)
    print("Test: ", test.shape)
    print("Canceled in train:", round(train["IsCanceled"].mean() * 100, 2), "%")
    print("Canceled in test: ", round(test["IsCanceled"].mean() * 100, 2), "%")