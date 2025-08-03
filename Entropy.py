import pandas as pd
import numpy as np
from antropy import perm_entropy, sample_entropy, spectral_entropy

# === Step 1: Load data ===
df = pd.read_excel("RASA_full_annotated_dataset.xlsx")
time_cols = ['0', '2', '5', '15', '30', '60']

# === Step 2: Ensure numeric and clean ===
df[time_cols] = df[time_cols].apply(pd.to_numeric, errors='coerce')
df_clean = df.dropna(subset=time_cols + ['Transformation']).copy()

# === Step 3: Compute entropy metrics per row ===
perm_entropies = []
samp_entropies = []
spec_entropies = []

for _, row in df_clean.iterrows():
    signal = row[time_cols].values.astype(float)

    # Permutation entropy
    pe = perm_entropy(signal, normalize=True)

    # Sample entropy
    se = sample_entropy(signal)

    # Spectral entropy (requires power spectral density)
    spe = spectral_entropy(signal, sf=1.0, method='fft', normalize=True)

    perm_entropies.append(pe)
    samp_entropies.append(se)
    spec_entropies.append(spe)

# === Step 4: Store results ===
df_clean['Permutation_Entropy'] = perm_entropies
df_clean['Sample_Entropy'] = samp_entropies
df_clean['Spectral_Entropy'] = spec_entropies

# === Step 5: Group summary by Transformation Class ===
summary = df_clean.groupby('Transformation')[[
    'Permutation_Entropy', 'Sample_Entropy', 'Spectral_Entropy'
]].agg(['mean', 'std', 'count']).round(4)

# === Step 6: Save to Excel ===
df_clean.to_excel("entropy_metrics_per_trajectory.xlsx", index=False)

# === Step 7: Display
print("📊 Entropy Metrics by Transformation Class:")
print(summary)
