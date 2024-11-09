# pip install scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample sentences
sentences = ["Sentence and document embeddings are fundamental to Natural Language Processing",
             "Sentence and document embeddings capture the meaning of text"]

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform sentences to create TF-IDF vectors
tfidf_vectors = vectorizer.fit_transform(sentences)

# Display results
print("Vocabulary:", vectorizer.vocabulary_)
print("TF-IDF Vectors:\n", tfidf_vectors.toarray())

# Output
# Vocabulary: {'sentence': 11, 'and': 0, 'document': 3, 'embeddings': 4, 'are': 1, 'fundamental': 5, 'to': 14, 'natural': 8, 'language': 6, 'processing': 10, 'capture': 2, 'the': 13, 'meaning': 7, 'of': 9, 'text': 12}
# TF-IDF Vectors:
#  [[0.25116439 0.35300279 0.         0.25116439 0.25116439 0.35300279
#   0.35300279 0.         0.35300279 0.         0.35300279 0.25116439
#   0.         0.         0.35300279]
#  [0.26844636 0.         0.37729199 0.26844636 0.26844636 0.
#   0.         0.37729199 0.         0.37729199 0.         0.26844636
#   0.37729199 0.37729199 0.        ]]