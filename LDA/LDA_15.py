import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

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
class LemmaTokenizer:

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc.lower()) if
                t not in Punctuation_Mark
                and t not in all_stopword
                and len(t) >= 2
                and not t.isdigit()
                and not t.replace('.', '', 1).isdigit()]


lemma_tokenizer = LemmaTokenizer()
text_processed = [lemma_tokenizer(doc) for doc in text]

print(f"Total documents: {len(text_processed)}")

# Create dictionary and corpus
dictionary = corpora.Dictionary(text_processed)
dictionary.filter_extremes(no_below=0.01, no_above=0.75)

corpus = [dictionary.doc2bow(doc) for doc in text_processed]

print(f"Dictionary size: {len(dictionary)}")

# LDA
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=15,
    passes=20,
    iterations=200,
    chunksize=2000,
    random_state=67,
    alpha='auto',
    eta='auto'
)

# Save the result
topic_word_matrix = []

for idx, topic in lda_model.show_topics(num_topics=15, num_words=10, formatted=False):
    for rank, (word, weight) in enumerate(topic, 1):
        topic_word_matrix.append({
            'Topic': idx,
            'Rank': rank,
            'Word': word,
            'Probability': round(weight, 4)
        })

topic_df = pd.DataFrame(topic_word_matrix)
topic_df.to_csv('Topic.csv', index=False)

print("Saved Topic.csv")

# pyLDAvis Visualization
vis = gensimvis.prepare(lda_model, corpus, dictionary)

pyLDAvis.save_html(vis, 'pyLDAvis.html')
print("Saved pyLDAvis.html")

# Most representative document for each topics
doc_topic_dist = list(lda_model[corpus])

# Store the most representative document for each topic
representative_doc = []

for topic_id in range(15):
    max_prob = -1
    best_doc_id = -1

    # Find document with highest probability for this topic
    for doc_id, topic_dist in enumerate(doc_topic_dist):
        # Get probability of current topic in this document
        prob = dict(topic_dist).get(topic_id, 0.0)

        if prob > max_prob:
            max_prob = prob
            best_doc_id = doc_id

    if best_doc_id != -1:
        representative_doc.append({
            'Topic': topic_id,
            'Max_Probability': round(max_prob, 4),
            'Document_ID': best_doc_id,
            'Abstract': text[best_doc_id]
        })

# Save the result
rep_df = pd.DataFrame(representative_doc)
rep_df.to_csv('Most_Representative_Document_Per_Topic.csv', index=False)
