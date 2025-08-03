import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Step 1: Load data ===
df = pd.read_excel("RASA_site_level_with_angles.xlsx")

# === Step 2: Global Commutator Test (Early vs Late) ===
df['T'] = df['Late_Mean'] - df['Early_Mean']
df['T_T'] = df['Late_Mean'] + df['T']
df['Delta_commute'] = (df['T_T'] - df['Early_Mean']).abs()

print("\n📊 Global Commutator Deviation (Early vs. Late):")
print(f"Mean:   {df['Delta_commute'].mean():.4f}")
print(f"Median: {df['Delta_commute'].median():.4f}")
print(f"Std:    {df['Delta_commute'].std():.4f}")

# === Step 3: Plot Global Histogram ===
plt.figure(figsize=(8, 5))
plt.hist(df['Delta_commute'], bins=50, color='steelblue', edgecolor='black')
plt.title("Deviation of T(T(ρ)) from Early — Commutator Error")
plt.xlabel("Absolute Deviation Magnitude")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("commutator_error_global.png", dpi=300)
plt.show()

# === Step 4: Class-wise Commutator Deviation ===
class_summary = df.groupby('Transformation')['Delta_commute'].agg(['mean', 'median', 'std', 'count']).round(4)
print("\n📚 Per-Class Commutator Deviation (Early vs. Late):")
print(class_summary)

# === Step 5: Now use raw columns '0' and '60' if available ===
if '0' in df.columns and '60' in df.columns:
    df['T_0_60'] = df['60'] - df['0']
    df['T_T_0_60'] = df['60'] + df['T_0_60']
    df['Delta_commute_0_60'] = (df['T_T_0_60'] - df['0']).abs()

    print("\n📊 Global Commutator Deviation (0 vs. 60):")
    print(f"Mean:   {df['Delta_commute_0_60'].mean():.4f}")
    print(f"Median: {df['Delta_commute_0_60'].median():.4f}")
    print(f"Std:    {df['Delta_commute_0_60'].std():.4f}")

    plt.figure(figsize=(8, 5))
    plt.hist(df['Delta_commute_0_60'], bins=50, color='darkorange', edgecolor='black')
    plt.title("Deviation of T(T(ρ)) from Timepoint 0 — Commutator Error")
    plt.xlabel("Absolute Deviation Magnitude")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("commutator_error_0_60.png", dpi=300)
    plt.show()

    # === Per-Class summary for raw 0 vs 60 ===
    class_summary_0_60 = df.groupby('Transformation')['Delta_commute_0_60'].agg(['mean', 'median', 'std', 'count']).round(4)
    print("\n📚 Per-Class Commutator Deviation (0 vs. 60):")
    print(class_summary_0_60)
else:
    print("\n⚠️ Raw columns '0' and '60' not found — skipping strict test.")
