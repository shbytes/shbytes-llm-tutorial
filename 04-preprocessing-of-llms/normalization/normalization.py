import re
import contractions
import nltk
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import unicodedata

# Initialization
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')

def normalize_text(text):
    # Lowercasing
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra whitespaces
    text = ' '.join(text.split())

    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Expand contractions
    text = contractions.fix(text)

    # Remove accents
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

    # Spelling correction
    text = str(TextBlob(text).correct())

    # Tokenize
    tokens = word_tokenize(text)

    # Join tokens back to string
    return ' '.join(tokens)

sample_text = "It's imporant to preprocess text for LLMs, isn't it? Removing noise, like stopwords and punctuation, helps models perform better!"
normalized_text = normalize_text(sample_text)
print(normalized_text)

# Output
# important preprocess text alms is not removing noise like stopford punctuation helps models perform better