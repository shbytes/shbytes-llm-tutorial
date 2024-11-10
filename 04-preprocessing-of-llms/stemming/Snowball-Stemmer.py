# pip install nltk

import nltk             # import the library
nltk.download('punkt')  # download resources - for tokenization
from nltk.tokenize import word_tokenize

text = "Tokenization in a major technique in NLP. Word embeddings help models learn languages"
tokens = word_tokenize(text)
print(tokens)
# Output => ['Tokenization', 'in', 'a', 'major', 'technique', 'in', 'NLP', '.', 'Word', 'embeddings', 'help', 'models', 'learn', 'languages']

from nltk.stem import SnowballStemmer # import SnowballStemmer for stemming process

snowball = SnowballStemmer("english")  # create SnowballStemmer object which takes language as an argument
stemmed_words = [snowball.stem(word) for word in tokens]
print(stemmed_words)
# Output => ['token', 'in', 'a', 'major', 'techniqu', 'in', 'nlp', '.', 'word', 'embed', 'help', 'model', 'learn', 'languag']