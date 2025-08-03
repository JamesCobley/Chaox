import pandas as pd
import numpy as np
from math import atan2, degrees

# === Load the dataset ===
df = pd.read_excel("aay7315_Data_file_S1.xlsx")
df.columns = df.columns.astype(str)

# === Define timepoints and compute early/late means ===
early_cols = ['0', '2', '5', '15']
late_cols = ['30', '60']
df[early_cols + late_cols] = df[early_cols + late_cols].apply(pd.to_numeric, errors='coerce')
df_clean = df.dropna(subset=early_cols + late_cols).copy()

df_clean['Site_ID'] = df_clean['Gene Name'].astype(str) + "_" + df_clean['Site'].astype(str)
df_clean['Early_Mean'] = df_clean[early_cols].mean(axis=1)
df_clean['Late_Mean'] = df_clean[late_cols].mean(axis=1)

# === Transformation classification ===
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
    lambda row: classify_transformation(row['Early_Mean'], row['Late_Mean']), axis=1)

# === Angular deviation function ===
def deflection_angle_from_identity(early, late):
    delta = late - early
    return abs(degrees(atan2(delta, 60)))

df_clean['Angle_Degrees'] = df_clean.apply(
    lambda row: deflection_angle_from_identity(row['Early_Mean'], row['Late_Mean']), axis=1)

# === (A) Within-Protein Analysis ===
within_protein = df_clean.groupby('Gene Name')['Angle_Degrees'].agg(['mean', 'std', 'count']).reset_index()
print("\n📊 Within-Protein Angular Variability:")
print(within_protein.sort_values('std', ascending=False).head())

# === (B) Within-Site (Location) Analysis ===
within_site = df_clean.groupby('Site')['Angle_Degrees'].agg(['mean', 'std', 'count']).reset_index()
print("\n📍 Within-Site Angular Variability:")
print(within_site.sort_values('std', ascending=False).head())

# === (C) Commutativity Check: Same transformation by protein vs. site? ===
site_class = df_clean.groupby('Site')['Transformation'].agg(lambda x: x.value_counts().idxmax())
gene_class = df_clean.groupby('Gene Name')['Transformation'].agg(lambda x: x.value_counts().idxmax())

# Create mergeable maps
df_clean['Protein_Class'] = df_clean['Gene Name'].map(gene_class)
df_clean['Site_Class'] = df_clean['Site'].map(site_class)

# Compare assignments
df_clean['Commutes'] = df_clean['Protein_Class'] == df_clean['Site_Class']
commute_rate = df_clean['Commutes'].mean()
print(f"\n🔁 Commutativity Rate between Protein and Site Class Assignments: {commute_rate:.2%}")
