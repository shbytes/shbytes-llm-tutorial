# pip install tensorflow_hub
# pip install tensorflow

import tensorflow as tf
import tensorflow_hub as hub

# Load Universal Sentence Encoder model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Example sentence
sentences = ["Sentence and document embeddings are fundamental to Natural Language Processing",
             "Sentence and document embeddings capture the meaning of text"]

# Generate sentence embedding
embedding = embed(sentences)

print("Embedding Shape:", embedding.shape)
print("Embedding:", embedding)

# Output
# Embedding Shape: (2, 512)
# Embedding: tf.Tensor(
# [[-0.03685547 -0.0234395   0.01266829 ...  0.05053782 -0.03162806
#   -0.0085871 ]
#  [ 0.01263339 -0.04275355  0.06820477 ...  0.04539199 -0.01522196
#    0.04380015]], shape=(2, 512), dtype=float32)