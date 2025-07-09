# Natural Language Processing (NLP) from Scratch

# -----------------------------
# 01. Tokenization
# -----------------------------
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize

text = """
Natural Language Processing (NLP) enables computers to understand human language.
This includes tasks like tokenization, stemming, and more.
"""

# Sentence Tokenization
sentences = sent_tokenize(text)
print("Sentences:", sentences)

# Word Tokenization
words = word_tokenize(text)
print("Words:", words)

# -----------------------------
# 02. Stemming
# -----------------------------
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

stemmed_words = [stemmer.stem(word) for word in words]
print("Stemmed Words:", stemmed_words)

# -----------------------------
# 03. Lemmatization
# -----------------------------
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
print("Lemmatized Words:", lemmatized_words)

# -----------------------------
# 04. Stopwords
# -----------------------------
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
print("Filtered Words:", filtered_words)

# -----------------------------
# 05. POS Tagging
# -----------------------------
nltk.download('averaged_perceptron_tagger')
pos_tags = nltk.pos_tag(filtered_words)
print("POS Tags:", pos_tags)

# -----------------------------
# 06. Named Entity Recognition
# -----------------------------
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk import ne_chunk

ner_tree = ne_chunk(pos_tags)
print("Named Entities:")
print(ner_tree)

# -----------------------------
# 07. Sentence Segmentation
# (Already done in 01 using sent_tokenize)

# -----------------------------
# 08. Bag of Words
# -----------------------------
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "NLP is fun and useful.",
    "We use NLP to process language."
]

vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(corpus)
print("BoW Vocabulary:", vectorizer.vocabulary_)
print("BoW Array:\n", bow.toarray())

# -----------------------------
# 09. TF-IDF
# -----------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus)
print("TF-IDF Array:\n", tfidf.toarray())

# -----------------------------
# 10. N-Grams
# -----------------------------
ngram_vectorizer = CountVectorizer(ngram_range=(1,2))
ngrams = ngram_vectorizer.fit_transform(corpus)
print("N-Gram Vocabulary:", ngram_vectorizer.vocabulary_)

# -----------------------------
# 11. Text Classification - Logistic Regression
# -----------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = tfidf
y = [1, 0]  # Just sample labels
model = LogisticRegression()
model.fit(X, y)
print("LogReg Prediction:", model.predict(X))

# -----------------------------
# 12. Text Classification - SVM
# -----------------------------
from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(X, y)
print("SVM Prediction:", svm_model.predict(X))

# -----------------------------
# 13. Word2Vec Word Embeddings
# -----------------------------
from gensim.models import Word2Vec

sample_sentences = [word_tokenize(s.lower()) for s in corpus]
w2v_model = Word2Vec(sample_sentences, vector_size=50, window=2, min_count=1, workers=4)
print("Word2Vec - vector for 'nlp':", w2v_model.wv['nlp'])

# -----------------------------
# 14. Text Emotion Detection (Example with hardcoded dataset)
# -----------------------------
train_texts = ["I am happy", "I feel sad", "This is exciting"]
train_labels = ["happy", "sad", "happy"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train_texts)
y = train_labels

clf = LogisticRegression()
clf.fit(X, y)
print("Emotion Prediction:", clf.predict(vectorizer.transform(["I am thrilled"])))

# -----------------------------
# 15. Topic Modelling with LDA
# -----------------------------
from gensim import corpora, models

lda_texts = ["NLP is amazing", "Deep learning in NLP", "AI and language"]
lda_tokens = [word_tokenize(doc.lower()) for doc in lda_texts]
dictionary = corpora.Dictionary(lda_tokens)
corpus_bow = [dictionary.doc2bow(text) for text in lda_tokens]

lda_model = models.LdaModel(corpus_bow, num_topics=2, id2word=dictionary, passes=10)

print("LDA Topics:")
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")
