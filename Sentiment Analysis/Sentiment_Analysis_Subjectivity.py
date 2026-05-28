import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the data
df = pd.read_csv('../Document_Processed.csv')

# Convert Serial to Date (1 = 1831-12-12)
start_date = pd.to_datetime('1831/12/12')
df['Date'] = start_date + pd.to_timedelta(df['Serial'] - 1, unit='D')

# Remove 0 values
df['Subjectivity'] = df['Subjectivity'].replace(0, None)

# Plot
fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
ax.plot(df['Date'], df['Subjectivity'], label='Subjectivity', color='blue')

# Title + label
plt.title("Subjectivity Over Time (1831–1836)")

ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
plt.xticks(rotation=45)

plt.ylabel("Score (0 to 1)")

# limit
plt.ylim((0, 1))

# Other
plt.legend()
plt.grid()
plt.tight_layout()

# Save the plot
plt.savefig('Fig/Subjectivity.png')
plt.show()
