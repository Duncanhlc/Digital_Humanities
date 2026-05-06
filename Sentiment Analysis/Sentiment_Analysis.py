import pandas as pd
from textblob import TextBlob

# Load the .csv file
df = pd.read_csv('../Text.csv', dtype={'Sea/Land/Harbour(0/1/2)': str})
df.columns = ['date', 'place', 'content', 'Sea/Land/Harbour(0/1/2)']


# Create sentiment_analyzer function
def sentiment_analyzer(text):
    text_str = str(text).strip()
    # Handle "N/A" or truly empty cells
    if text_str.upper() == 'N/A':
        return None

    analysis = TextBlob(text_str)
    return round(analysis.sentiment.polarity, 3), round(analysis.sentiment.subjectivity, 3)


# Store value
polarity_list = []
subjectivity_list = []
last_polarity = None
last_subjectivity = None

# Main
for idx, row in df.iterrows():
    content = row['content']

    # Check if Column C is empty/NaN
    if pd.isna(content) or str(content).strip() == "":
        polarity_list.append(None)
        subjectivity_list.append(None)
        continue  # Move to the next row immediately

    content_str = str(content).strip().lower()

    # Handle "skip" logic
    if content_str == 'skip':
        polarity_list.append(last_polarity)
        subjectivity_list.append(last_subjectivity)

    # Handle "N/A" and normal text
    else:
        pol, sub = sentiment_analyzer(content)
        polarity_list.append(pol)
        subjectivity_list.append(sub)

        # Update 'last' values only if we got a valid score (not N/A)
        if pol is not None:
            last_polarity = pol
            last_subjectivity = sub

# Apply results
df['polarity'] = polarity_list
df['subjectivity'] = subjectivity_list

# Save result
df.to_csv("Text_Processed.csv", index=False)
