import nltk   # nltk Python library
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt_tab')   # download punkt_tab library

# Given text
text = "Tokenization in a major technique in NLP. Let's learn!"

# Word tokenization
word_tokens = word_tokenize(text)   # Use method word_tokenize to get word tokens
print("Word Tokens:", word_tokens)
# Output => Word Tokens: ['Tokenization', 'in', 'a', 'major', 'technique', 'in', 'NLP', '.', 'Let', "'s", 'learn', '!']

# Sentence tokenization
sentence_tokens = sent_tokenize(text)  # Use method sent_tokenize to get sentence tokens
print("Sentence Tokens:", sentence_tokens)
# Output => Sentence Tokens: ['Tokenization in a major technique in NLP.', "Let's learn!"]
