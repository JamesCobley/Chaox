import pandas as pd
import numpy as np
from scipy.stats import zscore

def compute_LTT_metrics(df, timepoint_cols=['0', '2', '5', '15', '30', '60'], epsilon=1e-8):
    
    
    # Extract and clean oxidation matrix
    oxidation_matrix = df[timepoint_cols].apply(pd.to_numeric, errors='coerce')
    oxidation_matrix = oxidation_matrix.dropna()
    
    # Prepare metric storage
    D_sum = []
    D_max = []
    LTT_lambda = []

    for idx, row in oxidation_matrix.iterrows():
        trajectory = row.values.astype(float)
        x0 = trajectory[0]
        
        # --- Metric 1: Total absolute divergence from baseline ---
        d_sum = np.sum(np.abs(trajectory[1:] - x0))
        D_sum.append(d_sum)

        # --- Metric 2: Maximum absolute excursion from baseline ---
        d_max = np.max(np.abs(trajectory[1:] - x0))
        D_max.append(d_max)

        # --- Metric 3: Lyapunov-like log divergence from start to end ---
        d_start = np.abs(trajectory[-1] - x0)
        ltt_lam = (1 / (len(trajectory) - 1)) * np.log((d_start + epsilon) / (epsilon))
        LTT_lambda.append(ltt_lam)

    # Add metrics to the cleaned dataframe
    df_metrics = oxidation_matrix.copy()
    df_metrics['LTT_D_sum'] = D_sum
    df_metrics['LTT_D_max'] = D_max
    df_metrics['LTT_lambda'] = LTT_lambda

    return df_metrics


def classify_LTT(df_metrics, z_thresh=1.5):
    """
    Classifies peptides into LTT-high, LTT-stable, or LTT-intermediate 
    based on z-scored values of LTT metrics.
    
    Parameters:
        df_metrics: DataFrame output from compute_LTT_metrics
        z_thresh: threshold for z-score to determine 'high' or 'low' classification
    
    Returns:
        df_classified: original DataFrame with added 'LTT_Class' column
    """
    # Compute z-scores
    z_D_sum = zscore(df_metrics['LTT_D_sum'])
    z_D_max = zscore(df_metrics['LTT_D_max'])
    z_lambda = zscore(df_metrics['LTT_lambda'])
    
    # Apply classification rules
    classification = []

    for i in range(len(df_metrics)):
        high = (z_D_sum[i] > z_thresh) and (z_D_max[i] > z_thresh) and (z_lambda[i] > z_thresh)
        low = (z_D_sum[i] < -z_thresh) and (z_D_max[i] < -z_thresh) and (z_lambda[i] < -z_thresh)
        
        if high:
            classification.append('LTT-High')
        elif low:
            classification.append('LTT-Stable')
        else:
            classification.append('LTT-Intermediate')

    # Add classification to DataFrame
    df_classified = df_metrics.copy()
    df_classified['LTT_Class'] = classification

    return df_classified


# Load your peptide redox data
df = pd.read_excel('/content/aay7315_Data_file_S1.xlsx')
df.columns = df.columns.astype(str)  # 🔧 Ensures consistent column name types

# Run the full LTT pipeline
df_ltt = compute_LTT_metrics(df)
df_classified = classify_LTT(df_ltt)

# View results
df_classified['LTT_Class'].value_counts()
df_classified.head()

# Summary statistics
total_peptides = len(df_classified)
class_counts = df_classified['LTT_Class'].value_counts()
class_percent = (class_counts / total_peptides * 100).round(2)

print(f"Total peptides analyzed: {total_peptides}\n")

for cls in ['LTT-High', 'LTT-Intermediate', 'LTT-Stable']:
    count = class_counts.get(cls, 0)
    percent = class_percent.get(cls, 0.0)
    print(f"{cls}: {count} peptides ({percent}%)")
