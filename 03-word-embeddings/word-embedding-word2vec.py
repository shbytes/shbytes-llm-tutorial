from gensim.models import Word2Vec
import nltk   # nltk Python library
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt_tab')   # download punkt_tab library

# Given text
text = "Tokenization in a major technique in NLP. Word embeddings help models learn languages"
# Word tokenization
word_tokens = word_tokenize(text)   # Use method word_tokenize to get word tokens

# Train the Word2Vec model
skipgram_model = Word2Vec(sentences=[word_tokens],
                          vector_size=50,   # Dimensions of the word vectors
                          window=5,         # Maximum distance between the current and predicted word within a sentence
                          min_count=1,      # Ignores all words with a total frequency lower than this
                          sg=1)             # Skip-Gram model (1 for Skip-Gram, 0 for CBOW)

# Get the embedding for a specific word
vector = skipgram_model.wv['NLP']
print("Embedding for 'NLP':", vector)

# Finding similar words
similar_words = skipgram_model.wv.most_similar("NLP")
print("Words similar to 'NLP':", similar_words)

# Output
# Embedding for 'NLP': [-0.01427803  0.00248206 -0.01435343 -0.00448924  0.00743861  0.01166625
#   0.00239637  0.00420546 -0.00822078  0.01445067 -0.01261408  0.00929443
#  -0.01643995  0.00407294 -0.0099541  -0.00849538 -0.00621797  0.01131042
#   0.0115968  -0.0099493   0.00154666 -0.01699156  0.01561961  0.01851458
#  -0.00548466  0.00160045  0.0014933   0.01095577 -0.01721216  0.00116891
#   0.01373884  0.00446319  0.00224935 -0.01864431  0.01696473 -0.01252825
#  -0.00598475  0.00698757 -0.00154526  0.00282258  0.00356398 -0.0136578
#  -0.01944962  0.01808117  0.01239611 -0.01382586  0.00680696  0.00041213
#   0.00950749 -0.01423989]
# Words similar to 'NLP': [('major', 0.2373521625995636), ('.', 0.1845843642950058), ('learn', 0.13940522074699402), ('help', 0.10704469680786133), ('a', 0.04550706967711449), ('models', -0.010117169469594955), ('in', -0.0560765378177166), ('Tokenization', -0.06485531479120255), ('Word', -0.08925073593854904), ('embeddings', -0.10186848789453506)]