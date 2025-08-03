import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === Step 1: Load input data ===
df = pd.read_excel("RASA_Geometric_Energy_Trajectories.xlsx")
df.columns = df.columns.astype(str)

# === Step 2: Normalize energy terms and compute geometric entropy ===
# Avoid log(0) or division by zero with small epsilon
epsilon = 1e-8

# Replace any negative Morse energies with epsilon for stability
df['Morse_Energy_Pos'] = df['Morse_Energy'].clip(lower=epsilon)
df['Dirichlet_Energy_Pos'] = df['Dirichlet_Energy'].clip(lower=epsilon)
df['Triangle_AUC_Pos'] = df['Triangle_AUC'].clip(lower=epsilon)

# === Step 3: Define geometric entropy metric (log-proportional) ===
df['Geometric_Entropy'] = (
    np.log(df['Triangle_AUC_Pos']) +
    np.log(df['Morse_Energy_Pos']) -
    np.log(df['Dirichlet_Energy_Pos'])
)

# Optional: replace any non-finite values
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Geometric_Entropy'])

# === Step 4: Plot distribution per transformation class ===
plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=df,
    x="Geometric_Entropy",
    hue="Transformation",
    fill=True,
    alpha=0.3,
    linewidth=1.5,
)
plt.title("Geometric Entropy Distribution by Transformation Class")
plt.xlabel("Geometric Entropy")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
plt.savefig("geometric_entropy_distribution.png", dpi=300)
plt.show()

# === Step 5: Summary statistics per class ===
summary = df.groupby("Transformation")["Geometric_Entropy"].agg(["mean", "std", "count"]).round(4)
print("\n📊 Geometric Entropy by Transformation Class:")
print(summary)

# === Step 6: Export with new column ===
df.to_excel("RASA_Geometric_Energy_with_Entropy.xlsx", index=False)
print("\n✅ Exported: RASA_Geometric_Energy_with_Entropy.xlsx")
