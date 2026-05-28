import pandas as pd
from scipy.stats import f_oneway
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('../Document_Processed.csv')
category = df.columns['Sea/Land/Harbour(0/1/2)']

# Prepare data for each category
data = []

for cat in [0, 1, 2]:
    mask = df[category] == cat
    polarity = df.loc[mask, 'Polarity'].dropna()
    polarity = polarity[polarity != 0]

    data.append(polarity.values)
    print(f"Category {cat}: {len(polarity)} samples, Mean = {polarity.mean():.4f}, Std = {polarity.std():.4f}")

# Perform One-Way ANOVA
f_stat, p_value = f_oneway(*data)

print("\n" + "=" * 50)
print("ONE-WAY ANOVA RESULTS")
print("=" * 50)
print(f"F-statistic : {f_stat:.4f}")
print(f"p-value     : {p_value:.6f}")

if p_value < 0.05:
    print("✅ Result: Statistically significant difference between categories (p < 0.05)")
else:
    print("❌ Result: No statistically significant difference between categories")

# Plot
plt.figure(figsize=(10.8, 7.2), dpi=100)
plt.boxplot(data, labels=['Sea', 'Land', 'Harbour'])

# Title + label
plt.title('Polarity Distribution by Category')
plt.ylabel('Polarity Score')

# Other
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig("Boxplot.png")
plt.show()
