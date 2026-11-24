"""Clustering: group similar bookings (no target, no train/test split)."""

from src.data.split import SEED
from src.features.selection import CLUSTERING_FEATURES
from src.preprocessing.pipeline import build_preprocessor, load_prepared_data

df = load_prepared_data()
X = df[CLUSTERING_FEATURES]