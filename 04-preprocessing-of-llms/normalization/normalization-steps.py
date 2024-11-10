def lowercase_text(text):
    return text.lower()

def uppercase_text(text):
    return text.upper()

def titlecase_text(text):
    return text.title()

print(lowercase_text("Shbytes"))
print(uppercase_text("Shbytes"))
print(titlecase_text("shbytes"))

# Output
# shbytes
# SHBYTES
# Shbytes

print("....................................")

# Example Program - Punctuation Removal
import re

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

plain_text = remove_punctuation("Hello, world! This is an example sentence with punctuation: commas, periods, and exclamation marks!")
print(plain_text)

# Output
# Hello world This is an example sentence with punctuation commas periods and exclamation marks

print("....................................")

# Example Program - Removing Extra Whitespaces

def normalize_whitespace(text):
    return ' '.join(text.split())

text_with_whitespaces = "  This is an example     sentence with extra whitespaces  "
normalize_text = normalize_whitespace(text_with_whitespaces)
print(normalize_text)

# Output => This is an example sentence with extra whitespaces

print("....................................")

# Example Program - Stopword Removal
import nltk
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')

# stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

normalize_text = remove_stopwords("This is an example sentence with some stopwords")
print(normalize_text)

# Output => This example sentence stopwords

print("....................................")

# Example Program - Expanding Contractions
# pip install contractions
import contractions

def expand_contractions(text):
    return contractions.fix(text)

expanded_text = expand_contractions("LLMs aren't perfect, but they're helpful in NLP tasks")
print(expanded_text)

# Output => LLMs are not perfect, but they are helpful in NLP tasks

print("....................................")

# Example Program - Spelling Correction using textblob
# pip install textblob
from textblob import TextBlob

def correct_spelling(text):
    return str(TextBlob(text).correct())

corrected_text = correct_spelling("Ths is an exmple of usng library with NLP and LLMs.")
print(corrected_text)

# Output => The is an example of using library with NLP and LLMs.

print("....................................")

# Example Program - Spelling Correction using spellchecker
# pip install spellchecker
from spellchecker import SpellChecker

# Initialize the spell checker
spell = SpellChecker()

# Example text with spelling errors
text = "Ths is an exmple of usng library with NLP and LLMs"

# Split the text into words
words = text.split()

# Correct misspelled words
corrected_text = " ".join([spell.correction(word) if word in spell.unknown(words) else word for word in words])
print(corrected_text)

# Output => Ths is an example of using library with NLP and LLMs

print("....................................")

# Example Program - Handling Accents
# pip install unidecode
import re
from unidecode import unidecode

def remove_accent(text):
    # Convert accented characters to ASCII
    ascii_text = unidecode(text)
    return ascii_text;

def remove_special_characters(text):
    # Remove special characters (keeping only letters and spaces)
    simple_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return simple_text

# Sample text with special characters and accents
sample_text = "Café au lait: A popular drink in cafés across countries like México and España!!"
no_accent_text = remove_accent(sample_text)
# special characters from no_accent_text
normalized_text = remove_special_characters(no_accent_text)

print("Original Text:", sample_text)
print("No Accent Text:", no_accent_text)
print("Normalized Text:", normalized_text)

# Output
# Original Text: Café au lait: A popular drink in cafés across countries like México and España!!
# No Accent Text: Cafe au lait: A popular drink in cafes across countries like Mexico and Espana!!
# Normalized Text: Cafe au lait A popular drink in cafes across countries like Mexico and Espana

