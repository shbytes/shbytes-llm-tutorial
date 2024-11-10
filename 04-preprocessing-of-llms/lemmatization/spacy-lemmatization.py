# pip install spacy
import spacy

# before load, download spacy module - python -m spacy download en_core_web_sm

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = ("In NLP, lemmatization refines text by reducing words to their base forms. "
        "This help models to understand linguistic structures and contexts more accurately.")

# Process the text
doc = nlp(text)

# Extract lemmas
lemmatized_words = [token.lemma_ for token in doc]
print("Original words:", doc)
print("Lemmatized words:", lemmatized_words)

# Output
# Original words: In NLP, lemmatization refines text by reducing words to their base forms. This help models to understand linguistic structures and contexts more accurately.
# Lemmatized words: ['in', 'NLP', ',', 'lemmatization', 'refine', 'text', 'by', 'reduce', 'word', 'to', 'their', 'base', 'form', '.', 'this', 'help', 'model', 'to', 'understand', 'linguistic', 'structure', 'and', 'context', 'more', 'accurately', '.']