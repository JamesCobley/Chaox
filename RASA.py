import numpy as np
import pandas as pd

# === Step 1: Load the data ===
# Replace this with your file path or DataFrame input
df = pd.read_excel('/content/aay7315_Data_file_S1.xlsx')  # Or use pd.read_csv if needed
df.columns = df.columns.astype(str)  # Ensure column headers are strings

# === Step 2: Define early and late time windows ===
early_cols = ['0', '2', '5', '15']
late_cols = ['30', '60']
all_cols = early_cols + late_cols

# === Step 3: Ensure numeric values ===
df[all_cols] = df[all_cols].apply(pd.to_numeric, errors='coerce')

# === Step 4: Drop rows with missing data ===
df_clean = df.dropna(subset=all_cols).copy()

# === Step 5: Create Site_ID for grouping ===
df_clean['Site_ID'] = df_clean['Gene Name'].astype(str) + "_" + df_clean['Site'].astype(str)

# === Step 6: Compute early and late mean oxidation states ===
df_clean['Early_Mean'] = df_clean[early_cols].mean(axis=1)
df_clean['Late_Mean'] = df_clean[late_cols].mean(axis=1)

# === Step 7: Define transformation classification function ===
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

# === Step 8: Apply classification to each peptide ===
df_clean['Transformation'] = df_clean.apply(
    lambda row: classify_transformation(row['Early_Mean'], row['Late_Mean']),
    axis=1
)

# === Step 9: (Optional) Aggregate transformation per Site_ID ===
site_transforms = df_clean.groupby('Site_ID')['Transformation'].agg(lambda x: x.value_counts().idxmax())

# === Step 10: Summary output ===
print("Peptide-level transformation distribution:")
print(df_clean['Transformation'].value_counts(), "\n")

print("Site-level dominant transformation:")
print(site_transforms.value_counts())
