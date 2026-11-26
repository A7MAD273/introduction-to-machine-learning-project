"""Predict ADR for a new booking using the saved regression pipeline (no retraining)."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.features.selection import REGRESSION_FEATURES

SELECTED_JSON = Path("outputs") / "regression_selected.json"

# A sample booking known at confirmation time (all 20 model inputs).
SAMPLE_BOOKING = {
    "LeadTime": 60,
    "ArrivalDateMonth": "July",
    "ArrivalDateWeekNumber": 29,
    "ArrivalDateDayOfMonth": 15,
    "StaysInWeekendNights": 1,
    "StaysInWeekNights": 3,
    "Adults": 2,
    "Children": 0,
    "Babies": 0,
    "Meal": "BB",
    "MarketSegment": "Online TA",
    "IsRepeatedGuest": 0,
    "PreviousCancellations": 0,
    "PreviousBookingsNotCanceled": 0,
    "ReservedRoomType": "A",
    "DepositType": "No Deposit",
    "Agent": "9",
    "HasCompany": 0,
    "DaysInWaitingList": 0,
    "CustomerType": "Transient",
}


def load_selected_pipeline():
    """Load the deployed pipeline (preprocessing + model) chosen during model selection."""
    if not SELECTED_JSON.exists():
        raise FileNotFoundError(
            f"{SELECTED_JSON} not found. Run: python -m src.regression.final_fit")
    selected = json.loads(SELECTED_JSON.read_text(encoding="utf-8"))
    path = Path(selected["artifact"])
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.regression.final_fit")
    return selected, joblib.load(path)


def predict_adr(booking):
    """Return the predicted ADR (price per night) for one booking (a dict)."""
    missing = [c for c in REGRESSION_FEATURES if c not in booking]
    if missing:
        raise ValueError(f"Missing booking fields: {missing}")
    selected, pipeline = load_selected_pipeline()
    row = pd.DataFrame([booking])[REGRESSION_FEATURES]
    return float(pipeline.predict(row)[0]), selected


def main():
    parser = argparse.ArgumentParser(description="Predict ADR for a new booking.")
    parser.add_argument("--json", help="path to a JSON file with the 20 booking fields")
    args = parser.parse_args()

    if args.json:
        booking = json.loads(Path(args.json).read_text(encoding="utf-8"))
    else:
        booking = SAMPLE_BOOKING
        print("No --json given: using the built-in sample booking.")

    price, selected = predict_adr(booking)
    print(f"Model: {selected['selected_model']} (criterion: {selected['criterion']})")
    print(f"Predicted ADR: {price:.2f} per night")


if __name__ == "__main__":
    main()
