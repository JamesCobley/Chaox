import pandas as pd
import numpy as np
from numpy.linalg import norm
from math import acos, degrees

# === Step 1: Load your data ===
df = pd.read_excel('aay7315_Data_file_S1.xlsx')  # <-- Update path if needed
df.columns = df.columns.astype(str)

# === Step 2: Define time windows ===
early_cols = ['0', '2', '5', '15']
late_cols = ['30', '60']
all_cols = early_cols + late_cols

# === Step 3: Ensure numeric values ===
df[all_cols] = df[all_cols].apply(pd.to_numeric, errors='coerce')

# === Step 4: Drop rows with missing data ===
df_clean = df.dropna(subset=all_cols).copy()

# === Step 5: Create Site_ID ===
df_clean['Site_ID'] = df_clean['Gene Name'].astype(str) + "_" + df_clean['Site'].astype(str)

# === Step 6: Compute early/late mean oxidation ===
df_clean['Early_Mean'] = df_clean[early_cols].mean(axis=1)
df_clean['Late_Mean'] = df_clean[late_cols].mean(axis=1)

# === Step 7: Transformation classification ===
def classify_transformation(early, late, threshold=0.3):
    delta = late - early
    if np.isclose(delta, 0, atol=0.05):
        return 'Identity'
    elif abs(delta) <= threshold:
        return 'Scaling'
    elif (early < 0.3) and (late > 0.7):
        return 'Bifurcation'
    elif abs(delta) > threshold:
        return 'Deformation'
    else:
        return 'Unclassified'

df_clean['Transformation'] = df_clean.apply(
    lambda row: classify_transformation(row['Early_Mean'], row['Late_Mean']),
    axis=1
)

# === Step 8: Angular deviation from identity vector (1,1) ===
def redox_direction_angle(early, late):
    v = np.array([early, late])
    ref = np.array([1, 1])
    if np.allclose(v, 0):
        return 0.0
    cos_sim = np.dot(v, ref) / (norm(v) * norm(ref))
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return degrees(acos(cos_sim))

df_clean['Angle_Degrees'] = df_clean.apply(
    lambda row: redox_direction_angle(row['Early_Mean'], row['Late_Mean']),
    axis=1
)

# === Step 9: Aggregate per Site_ID ===
site_summary = df_clean.groupby('Site_ID').agg({
    'Gene Name': 'first',
    'Site': 'first',
    'Early_Mean': 'mean',
    'Late_Mean': 'mean',
    'Transformation': lambda x: x.value_counts().idxmax(),
    'Angle_Degrees': 'mean'
}).reset_index()

# === Step 10: Export ===
site_summary.to_excel('RASA_site_level_with_angles.xlsx', index=False)
print("✅ Exported: RASA_site_level_with_angles.xlsx")

# === Print counts per transformation class ===
print("\n🔢 Number of Sites in Each Transformation Class:")
print(site_summary['Transformation'].value_counts())

# === Compute angles in radians from degrees ===
site_summary['Angle_Radians'] = np.radians(site_summary['Angle_Degrees'])

# === Summary: average angle by class ===
angle_summary = site_summary.groupby('Transformation')[['Angle_Degrees', 'Angle_Radians']].agg(['mean', 'std', 'count']).round(3)

print("\n📐 Angular Deviation by Transformation Class:")
print(angle_summary
