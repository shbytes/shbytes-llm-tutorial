import nltk             # import the library
nltk.download('punkt')  # download resources - for tokenization
from nltk.tokenize import word_tokenize

text = "Tokenization in a major technique in NLP. Word embeddings help models learn languages"
tokens = word_tokenize(text)

from nltk.stem import PorterStemmer # import PorterStemmer for stemming process
from nltk.stem import SnowballStemmer # import SnowballStemmer for stemming process
from nltk.stem import LancasterStemmer # import LancasterStemmer for stemming process

# Apply each stemmer
porter_stemmed = [PorterStemmer().stem(word) for word in tokens]
snowball_stemmed = [SnowballStemmer("english").stem(word) for word in tokens]
lancaster_stemmed = [LancasterStemmer().stem(word) for word in tokens]

# Display the results
print("Original Tokens:", tokens)
print("Porter Stemmed:", porter_stemmed)
print("Snowball Stemmed:", snowball_stemmed)
print("Lancaster Stemmed:", lancaster_stemmed)

# Output
# Original Tokens: ['Tokenization', 'in', 'a', 'major', 'technique', 'in', 'NLP', '.', 'Word', 'embeddings', 'help', 'models', 'learn', 'languages']
# Porter Stemmed: ['token', 'in', 'a', 'major', 'techniqu', 'in', 'nlp', '.', 'word', 'embed', 'help', 'model', 'learn', 'languag']
# Snowball Stemmed: ['token', 'in', 'a', 'major', 'techniqu', 'in', 'nlp', '.', 'word', 'embed', 'help', 'model', 'learn', 'languag']
# Lancaster Stemmed: ['tok', 'in', 'a', 'maj', 'techn', 'in', 'nlp', '.', 'word', 'embed', 'help', 'model', 'learn', 'langu']