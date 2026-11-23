"""Feature lists for each task, from the leakage audit.

Prediction point (regression and classification): the moment the booking is confirmed.
Analysis point (clustering): booking time.
Reasons for every column are in the report (full decision table).

Not done here (it learns from data, so it belongs in each task's pipeline,
fitted on the training set only):
- Agent: fill missing with "None", group rare agents into "Other"
  (OneHotEncoder(min_frequency=..., handle_unknown="infrequent_if_exist")).
- Children: fill the 4 missing values with 0.
"""

from src.data.load_data import load_raw_data
from src.preprocessing.clean import clean_data

REGRESSION_TARGET = "ADR"
CLASSIFICATION_TARGET = "IsCanceled"

# Removed from regression and classification
LEAKAGE_COLUMNS = [
    "ReservationStatus",         # equals the label
    "ReservationStatusDate",     # set when the booking ends
    "AssignedRoomType",          # decided at arrival
    "BookingChanges",            # grows after booking
    "RequiredCarParkingSpaces",  # requested at arrival
    "TotalOfSpecialRequests",    # grows after booking
    "Country",                   # may be set or corrected at arrival
]
NOT_USEFUL_COLUMNS = [
    "ArrivalDateYear",           # does not repeat, can't generalize to future years
    "DistributionChannel",       # copy of MarketSegment
]

REGRESSION_FEATURES = [
    "LeadTime",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "MarketSegment",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "DepositType",
    "Agent",
    "HasCompany",
    "DaysInWaitingList",
    "CustomerType",
]

# Same as regression + the price (known at booking)
CLASSIFICATION_FEATURES = REGRESSION_FEATURES + ["ADR"]

CLUSTERING_FEATURES = [
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Meal",
    "MarketSegment",
    "ReservedRoomType",
    "Agent",
    "HasCompany",
    "CustomerType",
    "ADR",
]


def add_features(df):
    """Add engineered columns. Return a copy."""
    df = df.copy()
    df["HasCompany"] = df["Company"].notna().astype(int)  # Company is 95% empty
    return df


if __name__ == "__main__":
    df = add_features(clean_data(load_raw_data()))
    lists = {
        "Regression": REGRESSION_FEATURES,
        "Classification": CLASSIFICATION_FEATURES,
        "Clustering": CLUSTERING_FEATURES,
    }
    for name, cols in lists.items():
        missing = [c for c in cols if c not in df.columns]
        print(f"{name}: {len(cols)} features, missing columns: {missing}")

    assert REGRESSION_TARGET not in REGRESSION_FEATURES
    assert CLASSIFICATION_TARGET not in CLASSIFICATION_FEATURES
    assert CLASSIFICATION_TARGET not in CLUSTERING_FEATURES
    for cols in lists.values():
        assert not set(cols) & set(LEAKAGE_COLUMNS)
    print("All checks passed.")