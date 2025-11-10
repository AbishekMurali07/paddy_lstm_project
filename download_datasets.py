"""
download_datasets.py
Generates synthetic but realistic datasets for Tamil Nadu paddy yield prediction.
Creates: crop_data.csv, rainfall_data.csv, soil_data.csv, and final_tn_dataset.csv
"""

import pandas as pd
import numpy as np
import os

# -------------------- Configuration --------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Tamil Nadu districts where paddy is cultivated
districts = [
    "Thanjavur", "Tiruvarur", "Tiruvallur", "Kancheepuram",
    "Villupuram", "Cuddalore", "Tirunelveli"
]

years = list(range(2010, 2025))
np.random.seed(42)

# -------------------- Generate Crop Data --------------------
crop_data = []
for district in districts:
    for year in years:
        area = np.random.uniform(15000, 80000)  # hectares
        production = area * np.random.uniform(2.5, 4.8)  # tonnes
        yield_val = production / area  # tonnes per hectare
        crop_data.append([district, year, area, production, yield_val])

crop_df = pd.DataFrame(crop_data, columns=["District", "Year", "Area", "Production", "Yield"])
crop_df.to_csv(os.path.join(DATA_DIR, "crop_data.csv"), index=False)

# -------------------- Generate Rainfall Data --------------------
rainfall_data = []
for district in districts:
    for year in years:
        annual_rainfall = np.random.uniform(600, 1600)  # mm
        avg_temp = np.random.uniform(25, 33)  # °C
        rainfall_data.append([district, year, annual_rainfall, avg_temp])

rain_df = pd.DataFrame(rainfall_data, columns=["District", "Year", "Annual_Rainfall_mm", "Avg_Temperature_C"])
rain_df.to_csv(os.path.join(DATA_DIR, "rainfall_data.csv"), index=False)

# -------------------- Generate Soil Data --------------------
soil_types = ["Alluvial", "Red Loam", "Black Soil", "Laterite", "Clay"]
soil_data = []
for district in districts:
    soil_type = np.random.choice(soil_types)
    nitrogen = np.random.uniform(0.2, 0.6)
    phosphorus = np.random.uniform(0.1, 0.4)
    potassium = np.random.uniform(0.2, 0.5)
    soil_data.append([district, soil_type, nitrogen, phosphorus, potassium])

soil_df = pd.DataFrame(soil_data, columns=["District", "Soil_Type", "Nitrogen", "Phosphorus", "Potassium"])
soil_df.to_csv(os.path.join(DATA_DIR, "soil_data.csv"), index=False)

# -------------------- Merge Datasets --------------------
merged_df = pd.merge(crop_df, rain_df, on=["District", "Year"], how="left")
merged_df = pd.merge(merged_df, soil_df, on="District", how="left")

# Add small noise and compute a synthetic "Predicted_Yield" column
merged_df["Predicted_Yield"] = (
    merged_df["Yield"] +
    (merged_df["Annual_Rainfall_mm"] - 1000) * 0.0005 +
    (merged_df["Avg_Temperature_C"] - 28) * -0.05 +
    np.random.normal(0, 0.1, len(merged_df))
)

# Save final merged dataset
final_path = os.path.join(DATA_DIR, "final_tn_dataset.csv")
merged_df.to_csv(final_path, index=False)

print(f"✅ Synthetic datasets created successfully in '{DATA_DIR}/'")
print(f"📄 Files generated:")
print(f" - crop_data.csv ({len(crop_df)} rows)")
print(f" - rainfall_data.csv ({len(rain_df)} rows)")
print(f" - soil_data.csv ({len(soil_df)} rows)")
print(f" - final_tn_dataset.csv ({len(merged_df)} rows)")
