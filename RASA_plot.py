import matplotlib.pyplot as plt
import seaborn as sns
# Define early and late windows
early_cols = ['0', '2', '5', '15']
late_cols = ['30', '60']

# Compute early and late means
df['Early_Mean'] = df[early_cols].mean(axis=1)
df['Late_Mean'] = df[late_cols].mean(axis=1)

# Classifier function for time-domain algebra
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

# Apply classification to each row
df['Transformation'] = df.apply(
    lambda row: classify_transformation(row['Early_Mean'], row['Late_Mean']),
    axis=1
)

# Timepoints in minutes
timepoints = [0, 2, 5, 15, 30, 60]
timepoint_cols = ['0', '2', '5', '15', '30', '60']

# Ensure numeric
df[timepoint_cols] = df[timepoint_cols].apply(pd.to_numeric, errors='coerce')

# Drop NaNs
df_plot = df.dropna(subset=timepoint_cols + ['Transformation'])

# Melt to long format for plotting
df_melted = df_plot.melt(
    id_vars='Transformation',
    value_vars=timepoint_cols,
    var_name='Time',
    value_name='Oxidation'
)

# Convert time to numeric
df_melted['Time'] = df_melted['Time'].astype(str).astype(float)

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_melted,
    x='Time',
    y='Oxidation',
    hue='Transformation',
    errorbar='ci',  # show confidence interval
    estimator='mean',
    linewidth=2
)
plt.title("RASA Transformation Class")
plt.xlabel("Time (minutes)")
plt.ylabel("Delta redox (log₂ fold-change)")
plt.grid(True)
plt.tight_layout()
plt.legend(title='Algebraic Class')
plt.savefig('algebraic_classes.png',dpi=300)
plt.show()
