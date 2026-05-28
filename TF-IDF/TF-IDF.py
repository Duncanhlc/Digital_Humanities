import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the .csv
df = pd.read_csv('../Document_Processed.csv')
text = df['Content'].fillna("").astype(str).tolist()

# Stopwords
nltk_stopword = set(stopwords.words('english'))
custom_stopword = {}
all_stopword = nltk_stopword.union(custom_stopword)
Punctuation_Mark = [',', '.', ';', ':', '"', '``', "''", '`', '(', ')', '%', '%)', '±', '>', '<', '≈', '&']


# Lemma_Tokenizer
class lemma_tokenizer:

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc.lower()) if
                t not in Punctuation_Mark
                and t not in all_stopword
                and len(t) >= 2
                and not t.isdigit()
                and not t.replace('.', '', 1).isdigit()]


lemma_tokenizer = lemma_tokenizer()
text_processed = [lemma_tokenizer(doc) for doc in text]

print(f"Total documents: {len(text_processed)}")

text_joined = [' '.join(doc) for doc in text_processed]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    norm=None,  # Use the original TF-IDF formula
    min_df=0.01,  # Appear in 1% of the documents at least
    smooth_idf=True,  # Smooth idf weights by adding one to document frequencies
    sublinear_tf=False  # Replace tf with 1 + log(tf)
)

# Fit and transform
tfidf_matrix = vectorizer.fit_transform(text_joined)
feature_term = vectorizer.get_feature_names_out()

print(f"{len(feature_term)} feature names were found.")

# Create DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_term)

# Top terms by average TF-IDF
mean_tfidf = tfidf_df.mean().sort_values(ascending=False)
top20 = mean_tfidf.head(20)

# Save the result
mean_tfidf = pd.DataFrame({
    'Term': mean_tfidf.index,
    'Score': mean_tfidf.values
})

mean_tfidf.to_csv("Mean_TF-IDF.csv", index=False)

# Plot
plt.figure(figsize=(12.8, 7.2), dpi=100)
sns.barplot(x=top20.values, y=top20.index, palette='viridis')

# Title + label
plt.title('Top 20 Terms by Average TF-IDF Score')
plt.xlabel('Average TF-IDF Score')
plt.ylabel('')

# Add value labels
for i, v in enumerate(top20.values):
    plt.text(v + 0.001, i, f'{v:.4f}', va='center', weight="bold")

# Other
plt.grid(axis='x')
plt.tight_layout()

# Save the plot
plt.savefig('TF-IDF.png')
plt.show()

# Word Cloud
tfidf_dict = dict(zip(mean_tfidf['Term'], mean_tfidf['Score']))

word_cloud = WordCloud(
    width=1080, height=720,
    background_color='white',
    max_words=150,
    colormap='viridis',
    contour_width=3,
    contour_color='steelblue'
).generate_from_frequencies(tfidf_dict)

word_cloud.to_file("Word_cloud.png")
