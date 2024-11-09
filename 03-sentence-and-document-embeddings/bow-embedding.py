# pip install scikit-learn
from sklearn.feature_extraction.text import CountVectorizer

# Sample sentences
sentences = ["Sentence and document embeddings are fundamental to Natural Language Processing",
             "Sentence and document embeddings capture the meaning of text"]

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform sentences to create BoW vectors
bow_vectors = vectorizer.fit_transform(sentences)

# Display results
print("Vocabulary:", vectorizer.vocabulary_)
print("BoW Vectors:\n", bow_vectors.toarray())

# Output
# Vocabulary: {'sentence': 11, 'and': 0, 'document': 3, 'embeddings': 4, 'are': 1, 'fundamental': 5, 'to': 14, 'natural': 8, 'language': 6, 'processing': 10, 'capture': 2, 'the': 13, 'meaning': 7, 'of': 9, 'text': 12}
# BoW Vectors:
#  [[1 1 0 1 1 1 1 0 1 0 1 1 0 0 1]
#  [1 0 1 1 1 0 0 1 0 1 0 1 1 1 0]]