import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Load the .csv
df = pd.read_csv('../Text_Processed.csv')
category = df.columns[3]

# Create 3 separate plots
categories = [0, 1, 2]

for cat in categories:
    # Filter data for current category
    mask = df[category] == cat
    polarity = df.loc[mask, 'Polarity'].dropna()
    polarity = polarity[polarity != 0]  # Drop zeros as in your original code

    # Calculate mean and standard deviation
    mean = polarity.mean()
    std = polarity.std()

    # Plot
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)

    # Histogram
    plt.hist(polarity, bins=100, density=True, alpha=0.7, label='Data Distribution', color='skyblue', edgecolor='black')

    # Bell curve
    x = np.linspace(-1, 1, 200)
    p = norm.pdf(x, mean, std)
    plt.plot(x, p, 'r', label=f'Normal Dist. Curve\n($\mu={mean:.2f}$, $\sigma={std:.2f}$)')

    # Mean line
    plt.axvline(mean, color='black', linestyle='dashed', label=f'Mean: {mean:.2f}')

    # Title + label
    plt.title(f'Distribution of Subjectivity - Category {cat} (n={len(polarity)})')
    plt.xlabel('Score')
    plt.ylabel('Density')

    # Limit
    plt.xlim((-1, 1))

    # Other
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'Subjectivity_Distribution_Category_{cat}.png')
    plt.show()
