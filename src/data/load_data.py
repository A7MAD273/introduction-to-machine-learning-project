import numpy as np
import pandas as pd                      

from pathlib import Path
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "H2.csv"
EXPECTED_SHAPE = (79330, 31)


def load_raw_data(path=DATA_PATH):
    """Read H2.csv, clean the text values, and return a DataFrame."""
    df = pd.read_csv(path)                 

    text_columns = df.select_dtypes(exclude="number").columns
    for col in text_columns:
        df[col] = df[col].str.strip()  

    df = df.replace("NULL", np.nan)    

    if df.shape != EXPECTED_SHAPE:   
        raise ValueError(f"Expected {EXPECTED_SHAPE}, but got {df.shape}")

    return df                  


if __name__ == "__main__":
    df = load_raw_data()
    print(df.shape)
    pd.set_option("display.max_columns", None)
    print(df.head())
    print("Missing Agent:", df["Agent"].isna().sum())
    