import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
df = pd.read_excel('/content/aay7315_Data_file_S1.xlsx')

# Ensure column names are strings
df.columns = df.columns.astype(str)

# Define timepoint columns
timepoint_cols = ['0', '2', '5', '15', '30', '60']

# Extract and clean the oxidation matrix
oxidation_matrix = df[timepoint_cols].apply(pd.to_numeric, errors='coerce')
oxidation_matrix_clean = oxidation_matrix.dropna()

# Optionally limit number of peptides to plot for clarity (e.g., first 500)
N = 6000
subset = oxidation_matrix_clean.iloc[:N]

# Define timepoints
timepoints = np.array([0, 2, 5, 15, 30, 60])

# Plot
plt.figure(figsize=(12, 6))
for i in range(len(subset)):
    plt.plot(timepoints, subset.iloc[i], alpha=0.2, color='steelblue')

plt.xlabel("Time (minutes)")
plt.ylabel("Redox state (log₂ fold-change)")
plt.title(f"Redox Peptide Oxidation Trajectories")
plt.grid(True)
plt.tight_layout()
plt.savefig('peptide_trajectories.png', dpi=300)
plt.show()
