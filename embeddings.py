import re
import spacy
import numpy as np
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------
# CARGA
# -------------------

nlp = spacy.load("es_core_news_sm")

with open("Book.txt", "r", encoding="utf-8") as f:
    text = f.read()

# -------------------
# LIMPIEZA
# -------------------

text = text.lower()
text = re.sub(r"\d+", "", text)
text = re.sub(r"[^\w\s]", "", text)
text = re.sub(r"\s+", " ", text)

doc = nlp(text)

tokens = [
    token.lemma_
    for token in doc
    if not token.is_stop
    and not token.is_punct
    and len(token.text) > 2
]

with open("clean_book.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(tokens))

# -------------------
# TF-IDF + PCA
# -------------------

processed_text = " ".join(tokens)

tfidf = TfidfVectorizer(max_features=50)
X = tfidf.fit_transform([processed_text])

pca = PCA(n_components=2)
coords = pca.fit_transform(X.toarray())

plt.figure(figsize=(8,6))
plt.scatter(coords[:,0], coords[:,1])

plt.title("Espacio Vectorial TF-IDF")
plt.savefig("tfidf_space.png")
plt.close()

# -------------------
# WORD2VEC
# -------------------

sentences = [tokens]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

words = list(model.wv.index_to_key)[:50]

vectors = np.array([
    model.wv[word]
    for word in words
])

pca = PCA(n_components=2)

result = pca.fit_transform(vectors)

plt.figure(figsize=(10,8))

for i, word in enumerate(words):
    plt.scatter(result[i,0], result[i,1])
    plt.annotate(word,
                 (result[i,0], result[i,1]))

plt.title("Espacio Vectorial Word2Vec")
plt.savefig("word2vec_space.png")
plt.close()

print("Proceso completado.")