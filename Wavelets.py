import pandas as pd
import numpy as np
import pywt
from scipy.stats import entropy as shannon_entropy

# === Step 1: Load and clean data ===
df = pd.read_excel("aay7315_Data_file_S1.xlsx")
df.columns = df.columns.astype(str)
time_cols = ['0', '2', '5', '15', '30', '60']
df[time_cols] = df[time_cols].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=time_cols).copy()

# === Step 2: Site ID and transformation class ===
df['Site_ID'] = df['Gene Name'].astype(str) + "_" + df['Site'].astype(str)

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

df['Early_Mean'] = df[time_cols[:4]].mean(axis=1)
df['Late_Mean'] = df[time_cols[4:]].mean(axis=1)
df['Transformation'] = df.apply(lambda row: classify_transformation(row['Early_Mean'], row['Late_Mean']), axis=1)

# === Step 3: Wavelet-based metrics ===
def safe_entropy(signal):
    """Compute Shannon entropy of normalized squared signal."""
    power = np.square(signal)
    total_power = np.sum(power)
    if total_power == 0:
        return 0.0
    prob = power / total_power
    return -np.sum(prob * np.log2(prob + 1e-12))

def extract_wavelet_features(traj):
    """Decompose trajectory and return energy, entropy, and slope."""
    try:
        coeffs = pywt.wavedec(traj, 'db1', level=1)
        cA, cD = coeffs

        # Energy
        energy = np.sum(np.square(cA)) + np.sum(np.square(cD))

        # Entropy
        entropy_A = safe_entropy(cA)
        entropy_D = safe_entropy(cD)
        total_entropy = entropy_A + entropy_D

        # Slope
        slope = np.mean(np.abs(cD)) / (np.mean(np.abs(cA)) + 1e-8)

        return energy, total_entropy, slope
    except Exception:
        return np.nan, np.nan, np.nan

# Apply to each row
results = df[time_cols].apply(lambda row: extract_wavelet_features(row.values), axis=1)
df['Wavelet_Energy'], df['Wavelet_Entropy'], df['Wavelet_Slope'] = zip(*results)

# === Step 4: Group by transformation class and summarize ===
summary = df.groupby('Transformation')[['Wavelet_Energy', 'Wavelet_Entropy', 'Wavelet_Slope']] \
            .agg(['mean', 'std', 'count']).round(4)

# === Step 5: Export and print ===
df.to_excel("RASA_Wavelet_Features.xlsx", index=False)
print("✅ Exported: RASA_Wavelet_Features.xlsx\n")
print("📊 Wavelet Metrics by Transformation Class:")
print(summary)
