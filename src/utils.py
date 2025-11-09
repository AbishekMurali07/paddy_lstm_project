"""
utils.py
Utility functions for preprocessing, prediction, and feature merging.
"""

import pandas as pd
import numpy as np

def preprocess_input(area, production, rainfall, temperature, nitrogen, phosphorus, potassium):
    """Prepare input vector for model prediction."""
    return np.array([[area, production, rainfall, temperature, nitrogen, phosphorus, potassium]])

def merge_features(crop_df, rainfall_df, soil_df, districts):
    """Merge all datasets into one DataFrame."""
    merged = pd.merge(crop_df, rainfall_df, on=["District", "Year"], how="left")
    merged = pd.merge(merged, soil_df, on="District", how="left")
    merged = merged[merged["District"].isin(districts)]
    return merged
