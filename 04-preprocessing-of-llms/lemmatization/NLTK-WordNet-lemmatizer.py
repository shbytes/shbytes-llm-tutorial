from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import nltk
nltk.download('wordnet') # download resources - for tokenization

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Sample text
text = ("In NLP, lemmatization refines text by reducing words to their base forms. "
        "This help models to understand linguistic structures and contexts more accurately.")
# Tokenize the text
tokens = word_tokenize(text)

# Apply lemmatization
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
print("Original words:", tokens)
print("Lemmatized words:", lemmatized_words)

# Output
# Original words: ['In', 'NLP', ',', 'lemmatization', 'refines', 'text', 'by', 'reducing', 'words', 'to', 'their', 'base', 'forms', '.', 'This', 'help', 'models', 'to', 'understand', 'linguistic', 'structures', 'and', 'contexts', 'more', 'accurately', '.']
# Lemmatized words: ['In', 'NLP', ',', 'lemmatization', 'refines', 'text', 'by', 'reducing', 'word', 'to', 'their', 'base', 'form', '.', 'This', 'help', 'model', 'to', 'understand', 'linguistic', 'structure', 'and', 'context', 'more', 'accurately', '.']
