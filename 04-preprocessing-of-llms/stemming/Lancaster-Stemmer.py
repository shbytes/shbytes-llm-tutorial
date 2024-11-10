# pip install nltk

import nltk             # import the library
nltk.download('punkt')  # download resources - for tokenization
from nltk.tokenize import word_tokenize

text = "Tokenization in a major technique in NLP. Word embeddings help models learn languages"
tokens = word_tokenize(text)
print(tokens)
# Output => ['Tokenization', 'in', 'a', 'major', 'technique', 'in', 'NLP', '.', 'Word', 'embeddings', 'help', 'models', 'learn', 'languages']

from nltk.stem import LancasterStemmer # import LancasterStemmer for stemming process

lancaster = LancasterStemmer()  # create LancasterStemmer object which takes language as an argument
stemmed_words = [lancaster.stem(word) for word in tokens]
print(stemmed_words)
# Output => ['tok', 'in', 'a', 'maj', 'techn', 'in', 'nlp', '.', 'word', 'embed', 'help', 'model', 'learn', 'langu']
