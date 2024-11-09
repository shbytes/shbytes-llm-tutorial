from gensim.models import FastText
import nltk   # nltk Python library
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt_tab')   # download punkt_tab library

# Given text
text = "Tokenization in a major technique in NLP. Word embeddings help models learn languages"
# Word tokenization
word_tokens = word_tokenize(text)   # Use method word_tokenize to get word tokens

# Train FastText model
model = FastText(sentences=word_tokens,
                 vector_size=50,   # Dimensions of the word vectors
                 window=3,         # Maximum distance between the current and predicted word within a sentence
                 min_count=1)      # Ignores all words with a total frequency lower than this

# Get embedding for a word
vector = model.wv['NLP']
print("Embedding for 'NLP':", vector)

# FastText can generate embeddings for out-of-vocabulary words
oov_vector = model.wv['NLPlang']
print("Embedding for 'NLPlang':", oov_vector)

# Embedding for 'NLP': [ 6.7112967e-04 -5.7537104e-03 -3.8209217e-04  1.8261531e-03
#  -4.3684859e-03 -2.1409548e-03 -4.6887714e-04  3.7225138e-03
#  -4.4143177e-03 -6.5339045e-03 -6.6128164e-03  1.6831801e-03
#  -2.9464439e-03  8.4889242e-03  9.8627678e-04 -1.1673558e-02
#   7.0546214e-03  1.0250329e-02 -3.7461389e-03  2.2268505e-03
#  -5.7210624e-03 -6.2403362e-03 -6.7902668e-03 -7.2669764e-03
#   3.5307240e-03  8.6204302e-05  1.4065602e-03 -2.6211115e-03
#   4.1991104e-03  1.6430151e-03 -8.8640861e-03 -4.2608319e-04
#  -2.8893941e-03 -8.8902814e-03  8.5078767e-03  8.2775457e-03
#  -1.3053512e-03 -7.5113936e-03 -6.6510378e-04 -5.4182769e-03
#  -1.8815467e-03 -1.6216278e-03 -4.9716611e-03 -1.7002182e-03
#  -2.9542008e-03 -5.1571857e-03 -2.6930883e-04  3.6215771e-03
#   1.0876763e-03 -4.1648340e-03]
# Embedding for 'NLPlang': [-4.45177453e-03  2.80638388e-03 -2.81983591e-03  1.83598965e-03
#  -9.30043927e-04  6.45409513e-04  1.56107056e-03 -1.67246442e-04
#   1.18434546e-04  8.74134188e-04 -1.01212325e-04 -3.30892578e-03
#   1.03953527e-03  2.51525576e-04  2.34705932e-03  1.38890883e-03
#  -5.85419533e-04  6.07060315e-03 -1.47976016e-03  1.72582723e-03
#   2.58205528e-03 -5.39712422e-03 -2.09404016e-03 -4.82992531e-04
#   7.34058442e-04  8.99983497e-05  2.25677999e-04 -2.43995848e-04
#  -1.77256425e-03  2.61991314e-04  1.97137496e-03 -2.31975014e-03
#   2.54085008e-03  7.32400105e-04  9.11888725e-04  6.78675307e-04
#  -2.13333569e-03 -1.84630568e-03  1.05537695e-03  9.96632152e-04
#   1.53413892e-03 -3.52063309e-03  4.74144612e-03 -2.25093812e-04
#  -4.46041767e-03  1.32573955e-03  1.13292388e-03 -1.61171251e-03
#   1.69862364e-03  8.82553868e-04]