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

from math import atan2, degrees

def deflection_angle_from_identity(early, late):
    """
    Compute the angular deflection (in degrees) between the flat trajectory
    (no change over time) and the observed redox shift from timepoint 0 to 60.
    
    The baseline vector is (0, 0) → (60, 0)
    The observed vector is (0, 0) → (60, late - early)
    The angle is arctangent((Δ redox) / Δ time), converted to degrees.
    
    Returns:
        Absolute angular deflection in degrees.
    """
    delta = late - early          # Redox shift (vertical change)
    time_interval = 60            # Minutes between t=0 and t=60
    angle_rad = atan2(delta, time_interval)
    return abs(degrees(angle_rad))  # Return positive deflection from flat
df_clean['Angle_Degrees'] = df_clean.apply(
    lambda row: deflection_angle_from_identity(row['Early_Mean'], row['Late_Mean']),
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
print(angle_summary)

# Group angular values per transformation class
auc_per_class = df_clean.groupby('Transformation')['Angle_Degrees'].sum().round(3)

# Optionally normalize by number of sites (mean area per site)
normalized_auc = df_clean.groupby('Transformation')['Angle_Degrees'].mean().round(3)

# Print AUC and normalized AUC
print("📐 AUC of Angular Deflection per Transformation Class (Degrees):")
print(auc_per_class)

print("\n📐 Normalized AUC (Mean Angular Deflection per Site):")
print(normalized_auc)
