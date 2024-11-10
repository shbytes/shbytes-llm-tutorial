# import libraries required for tokenization
import nltk
from nltk.tokenize import word_tokenize

# Sample text
text = ("In NLP, lemmatization refines text by reducing words to their base forms. "
        "This help models to understand linguistic structures and contexts more accurately.")

# Tokenize and lemmatize with POS
tokens = word_tokenize(text)

# import libraries required for Part of Speech (POS) lemmatization
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger_eng')

# Function to convert POS tags to WordNet format
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]
print("Original words:", tokens)
print("Lemmatized words:", lemmatized_words)

# Output
# Original words: ['In', 'NLP', ',', 'lemmatization', 'refines', 'text', 'by', 'reducing', 'words', 'to', 'their', 'base', 'forms', '.', 'This', 'help', 'models', 'to', 'understand', 'linguistic', 'structures', 'and', 'contexts', 'more', 'accurately', '.']
# Lemmatized words: ['In', 'NLP', ',', 'lemmatization', 'refines', 'text', 'by', 'reduce', 'word', 'to', 'their', 'base', 'form', '.', 'This', 'help', 'model', 'to', 'understand', 'linguistic', 'structure', 'and', 'context', 'more', 'accurately', '.']
