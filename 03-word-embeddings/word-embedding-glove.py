import numpy as np

# Load pre-trained GloVe embeddings (downloaded from a source)
embedding_dict = {}

# Download GloVe enbeddings from https://nlp.stanford.edu/projects/glove/
with open("glove.6B.100d.txt", encoding="utf8") as f: # Read embeddings as key-value pair dictionary
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embedding_dict[word] = vector

# Accessing the embedding for a specific word
print("Embedding for 'language':", embedding_dict.get("language"))

# Calculating similarity between words
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

similarity = cosine_similarity(embedding_dict["king"], embedding_dict["queen"])
print("Cosine similarity between 'king' and 'queen':", similarity)

# Output
# Embedding for 'language': [ 0.18519   0.34111   0.36097   0.27093  -0.031335  0.83923  -0.50534
#  -0.80062   0.40695   0.82488  -0.98239  -0.6354   -0.21382   0.079889
#  -0.29557   0.17075   0.17479  -0.74214  -0.2677    0.21074  -0.41795
#   0.027713  0.71123   0.2063   -0.12266  -0.80088   0.22942   0.041037
#  -0.56901   0.097472 -0.59139   1.0524   -0.66803  -0.70471   0.69757
#  -0.11137  -0.27816   0.047361  0.020305 -0.184    -1.0254    0.11297
#  -0.79547   0.41642  -0.2508   -0.3188    0.37044  -0.26873  -0.36185
#  -0.096621 -0.029956  0.67308   0.53102   0.62816  -0.11507  -1.5524
#  -0.30628  -0.4253    1.8887    0.3247    0.60202   0.81163  -0.46029
#  -1.4061    0.80229   0.2019    0.60938   0.063545  0.21925  -0.043372
#  -0.36648   0.61308   1.0207   -0.39014   0.1717    0.61272  -0.80342
#   0.71295  -1.0938   -0.50546  -0.99668  -1.6701   -0.31804  -0.62934
#  -2.0226    0.79405  -0.16994  -0.37627   0.57998   0.16643   0.1356
#   0.0943   -0.24154   0.7123   -0.4201    0.24735  -0.94449  -1.0794
#   0.3413    0.34704 ]
# Cosine similarity between 'king' and 'queen': 0.750769