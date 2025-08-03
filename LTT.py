import pandas as pd
import numpy as np
from scipy.stats import zscore
from math import atan2, degrees

# === Load your redox dataset ===
df = pd.read_excel("aay7315_Data_file_S1.xlsx")
df.columns = df.columns.astype(str)

# === Define time windows ===
early_cols = ['0', '2', '5', '15']
late_cols = ['30', '60']
timepoint_cols = early_cols + late_cols

# === Ensure numeric data and drop missing ===
df[timepoint_cols] = df[timepoint_cols].apply(pd.to_numeric, errors='coerce')
df_clean = df.dropna(subset=timepoint_cols).copy()

# === Create Site ID and compute early/late means ===
df_clean['Site_ID'] = df_clean['Gene Name'].astype(str) + "_" + df_clean['Site'].astype(str)
df_clean['Early_Mean'] = df_clean[early_cols].mean(axis=1)
df_clean['Late_Mean'] = df_clean[late_cols].mean(axis=1)

# === Assign transformation classes using RASA ===
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

# === Compute angular deflection (for context) ===
def deflection_angle_from_identity(early, late):
    delta = late - early
    time_interval = 60
    angle_rad = atan2(delta, time_interval)
    return abs(degrees(angle_rad))

df_clean['Angle_Degrees'] = df_clean.apply(
    lambda row: deflection_angle_from_identity(row['Early_Mean'], row['Late_Mean']),
    axis=1
)

# === Compute Lyapunov-like Trajectory Tracker (LTT) metrics ===
def compute_LTT_metrics(df, timepoint_cols, epsilon=1e-8):
    oxidation_matrix = df[timepoint_cols].dropna()
    D_sum, D_max, LTT_lambda = [], [], []

    for _, row in oxidation_matrix.iterrows():
        trajectory = row.values.astype(float)
        x0 = trajectory[0]
        D_sum.append(np.sum(np.abs(trajectory[1:] - x0)))
        D_max.append(np.max(np.abs(trajectory[1:] - x0)))
        d_start = np.abs(trajectory[-1] - x0)
        LTT_lambda.append((1 / (len(trajectory) - 1)) * np.log((d_start + epsilon) / epsilon))

    df_metrics = oxidation_matrix.copy()
    df_metrics['LTT_D_sum'] = D_sum
    df_metrics['LTT_D_max'] = D_max
    df_metrics['LTT_lambda'] = LTT_lambda
    return df_metrics

# Run LTT metrics
df_ltt = compute_LTT_metrics(df_clean, timepoint_cols)

# Merge with transformation labels
df_ltt['Site_ID'] = df_clean.loc[df_ltt.index, 'Site_ID']
df_ltt['Transformation'] = df_clean.loc[df_ltt.index, 'Transformation']

# === Summarize LTT metrics per transformation class ===
summary = df_ltt.groupby('Transformation')[['LTT_D_sum', 'LTT_D_max', 'LTT_lambda']]\
                .agg(['mean', 'std', 'count']).round(3)

print("📊 LTT Metrics by Transformation Class:")
print(summary)

# === Optional: Save outputs ===
df_ltt.to_excel("LTT_metrics_by_site.xlsx", index=False)
summary.to_excel("LTT_summary_by_class.xlsx")
print("\n✅ Exported: LTT_metrics_by_site.xlsx & LTT_summary_by_class.xlsx")
