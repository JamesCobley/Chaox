import pandas as pd
import numpy as np
from numpy import log2, sqrt, arccos, clip
import matplotlib.pyplot as plt

# === Load the data ===
df = pd.read_excel("RASA_site_level_with_angles.xlsx")

# === Functions ===
def shannon_entropy(values, bins=50):
    """
    Compute Shannon entropy of a distribution over bins as a discrete PMF.
    Returns entropy in bits.
    """
    values = values.dropna()
    hist, _ = np.histogram(values, bins=bins, range=(min(values), max(values)), density=False)
    pmf = hist / np.sum(hist)
    pmf = pmf[pmf > 0]
    return -np.sum(pmf * np.log2(pmf))


def fisher_information_metric(values, bins=50):
    values = values.dropna()
    hist, _ = np.histogram(values, bins=bins, range=(min(values), max(values)), density=True)
    hist = hist / np.sum(hist)
    grad = np.gradient(hist)
    fim = np.sum((grad**2) / (hist + 1e-10))  # prevent divide by zero
    return fim

def histogram_distribution(values, bins=50, range_max=None):
    hist, _ = np.histogram(values.dropna(), bins=bins, range=(0, range_max), density=True)
    return hist / np.sum(hist)

def fisher_rao_distance(P, Q):
    bc = np.sum(np.sqrt(P * Q))  # Bhattacharyya coefficient
    bc = clip(bc, 0, 1)
    return 2 * arccos(bc)

# === Compute metrics per class ===
bins = 50
angle_max = df['Angle_Degrees'].max()
results = []

for cls in df['Transformation'].unique():
    subset = df[df['Transformation'] == cls]
    angles = subset['Angle_Degrees']

    entropy = shannon_entropy(angles, bins=bins)
    fim = fisher_information_metric(angles, bins=bins)
    hist = histogram_distribution(angles, bins=bins, range_max=angle_max)

    results.append({
        'Transformation': cls,
        'Shannon_Entropy': entropy,
        'Fisher_Information': fim,
        'Histogram': hist
    })

# === Create summary DataFrame ===
summary_df = pd.DataFrame({
    'Transformation': [r['Transformation'] for r in results],
    'Shannon_Entropy': [r['Shannon_Entropy'] for r in results],
    'Fisher_Information': [r['Fisher_Information'] for r in results]
}).set_index('Transformation')

# === Compute pairwise Fisher–Rao distances ===
classes = summary_df.index.tolist()
fr_dist_matrix = pd.DataFrame(index=classes, columns=classes, dtype=float)

for i, r1 in enumerate(results):
    for j, r2 in enumerate(results):
        if i == j:
            fr_dist_matrix.loc[r1['Transformation'], r2['Transformation']] = 0.0
        elif pd.isna(fr_dist_matrix.loc[r1['Transformation'], r2['Transformation']]):
            dist = fisher_rao_distance(r1['Histogram'], r2['Histogram'])
            fr_dist_matrix.loc[r1['Transformation'], r2['Transformation']] = dist
            fr_dist_matrix.loc[r2['Transformation'], r1['Transformation']] = dist

# === Output ===
print("\n📊 Information Summary per Transformation Class:")
print(summary_df.round(4))

print("\n📐 Fisher–Rao Distance Matrix:")
print(fr_dist_matrix.round(4))

# === Optional: Export ===
summary_df.to_csv("RASA_information_summary.csv")
fr_dist_matrix.to_csv("RASA_fisher_rao_distances.csv")

print("\n✅ Exported: RASA_information_summary.csv and RASA_fisher_rao_distances.csv")
