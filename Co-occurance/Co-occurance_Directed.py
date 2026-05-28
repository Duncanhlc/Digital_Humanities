import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import networkx as nx

# Load the .csv
df = pd.read_csv('../Document_Processed.csv')
text = df['Content'].fillna("").astype(str).tolist()

# Stopwords
nltk_stopword = set(stopwords.words('english'))
custom_stopword = {'skip', '¡x', 'one', 'two', 'three', 'four', 'five', "day", "mr", "r.", "u",
                   "large", "great", "much", "little", "many", "would", "may", "every", "yet", "must"}
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

# Build co-occurrence matrix
edge_dict = defaultdict(int)
window_size = 3

for token_compared in text_processed:
    for i, word in enumerate(token_compared):
        # Look at words within the window (after current word)
        for j in range(i + 1, min(i + window_size + 1, len(token_compared))):
            w1 = word
            w2 = token_compared[j]
            # No longer sorting! We keep (source, target) order.
            edge_dict[(w1, w2)] += 1

# Create edge dataframe
edges = []
min_weight = 10

for (w1, w2), weight in edge_dict.items():
    if weight >= min_weight:
        edges.append([w1, w2, weight])

edge_df = pd.DataFrame(edges, columns=['Source', 'Target', 'Weight'])
edge_df = edge_df.sort_values('Weight', ascending=False)

print(f"\nTotal edges: {len(edge_df)}")

# Save the result
edge_df.to_csv('Edge_Directed.csv', index=False)

# Create and export .gexf for Gephi
graph = nx.DiGraph()

for _, row in edge_df.iterrows():
    graph.add_edge(row['Source'], row['Target'], weight=int(row['Weight']))

nx.write_gexf(graph, "Co-occurrence_Directed.gexf")
