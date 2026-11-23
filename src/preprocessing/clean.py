from src.data.load_data import load_raw_data


def clean_data(df):
    """Remove duplicate and invalid bookings. Return a cleaned copy."""
    df = df.copy()

    df = df.drop_duplicates()

    guests = df["Adults"] + df["Children"].fillna(0) + df["Babies"]
    nights = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    valid = (guests > 0) & (nights > 0) & (df["ADR"] <= 1000)
    df = df[valid]

    return df.reset_index(drop=True)


if __name__ == "__main__":
    raw = load_raw_data()
    clean = clean_data(raw)
    print("Before:", raw.shape)
    print("After: ", clean.shape)
    print("Removed:", len(raw) - len(clean))