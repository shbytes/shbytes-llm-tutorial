# pip install sentence_transformers
# pip install tf-keras==2.16.0 --no-dependencies
# pip install tensorflow-cpu==2.16.1
from sentence_transformers import SentenceTransformer

# Load Universal Sentence Encoder
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Sample sentences
sentences = ["Sentence and document embeddings are fundamental to Natural Language Processing",
             "Sentence and document embeddings capture the meaning of text"]

# Generate sentence embeddings
embeddings = model.encode(sentences)
print("USE Embeddings:\n", embeddings)

# Output
# USE Embeddings:
#  [[ 2.64762118e-02 -1.29387714e-02  8.34194273e-02  1.39148422e-02
#    4.49235030e-02  5.88709787e-02  8.17405060e-03  1.82299055e-02
#    5.51757738e-02 -5.31634875e-02  1.33789070e-02  2.63189133e-02
#    6.96781501e-02  2.17816774e-02 -1.43869044e-02  4.22834419e-02
# ......
# ......
#    2.55598258e-02  6.41460344e-02 -1.05694152e-01 -3.66729945e-02
#    9.54475775e-02  1.33654801e-02  4.04356010e-02 -3.36793028e-02
#    4.95735593e-02 -5.67018799e-03 -2.04745363e-02 -2.92272698e-02
#    1.37664704e-02  5.47013171e-02  3.32472324e-02 -1.30022531e-02
#    2.47585531e-02  1.99648552e-02 -2.18799971e-02  3.97324940e-04
#    1.80490538e-02  6.09547049e-02  9.78075042e-02 -2.97105499e-02]]