import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Load the .csv file
df = pd.read_csv('../Document_Processed.csv')
polarity = df['Polarity'].dropna()
polarity = polarity[polarity != 0]

# Calculate mean and standard deviation
mean, std = polarity.mean(), polarity.std()

# Plot
fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)

# Histogram
count, bins, ignored = plt.hist(polarity, bins=100, density=True, alpha=0.7, label='Data Distribution', color='skyblue', edgecolor='black')

# Bell Curve
x = np.linspace(-1, 1, 200)
p = norm.pdf(x, mean, std)

plt.plot(x, p, 'r', label=f'Normal Dist. Curve\n($\mu={mean:.2f}$, $\sigma={std:.2f}$)')

# Mean line
plt.axvline(mean, color='black', linestyle='dashed', label=f'Mean: {mean:.2f}')

# Title + label
plt.title('Distribution of Polarity')
plt.xlabel('Score')
plt.ylabel('Density')

# Limit
plt.xlim((-1, 1))

# Other
plt.legend()
plt.grid()
plt.tight_layout()

# Save the plot
plt.savefig('Fig/Polarity_Distribution.png')
plt.show()
