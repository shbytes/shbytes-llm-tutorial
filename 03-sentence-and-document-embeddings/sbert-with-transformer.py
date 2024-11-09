# pip install sentence_transformers
# pip install tf-keras==2.16.0 --no-dependencies
# pip install tensorflow-cpu==2.16.1
from sentence_transformers import SentenceTransformer

# Load Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Sample sentences
sentences = ["Sentence and document embeddings are fundamental to Natural Language Processing",
             "Sentence and document embeddings capture the meaning of text"]

# Generate sentence embeddings
sentence_embeddings = model.encode(sentences)
print("SBERT Embeddings:\n", sentence_embeddings)

# Output
# SBERT Embeddings:
#  [[-2.72467405e-01  6.55197501e-02 -9.68914405e-02 -2.89553314e-01
#    3.44820827e-01 -2.67881304e-01 -1.62187889e-01 -3.61999929e-01
#    1.37213007e-01  3.67059916e-01 -8.67417082e-02  3.54015231e-02
#    5.43699563e-01 -4.56121042e-02  1.12542279e-01  1.11673616e-01
#    3.48674476e-01  4.24488872e-01 -5.75616896e-01 -5.10968983e-01
# ......
# ......
#    4.44245994e-01 -3.22503656e-01  2.43170336e-01 -3.25154573e-01
#    2.40026981e-01  2.33099937e-01  1.39374211e-01 -2.95164883e-01
#    1.08170450e-01  9.97239575e-02  4.27583307e-01  2.24687885e-02
#   -2.58610658e-02  2.58651972e-01  6.08575881e-01 -2.47781843e-01]]