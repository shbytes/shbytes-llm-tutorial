import re

class RegexStemmer:
    def __init__(self):
        # Define regular expression patterns for common suffixes
        self.suffix_patterns = [
            (r'ization$', 'ize'),
            (r'iveness$', 'ive'),
            (r'es$', 'e'),
            (r'els$', 'el'),
            (r'dings$', ''),
        ]

    def stem(self, word):
        """Apply regex-based stemming to the given word."""
        for pattern, replacement in self.suffix_patterns:
            if re.search(pattern, word):
                return re.sub(pattern, replacement, word)
        return word  # Return the word unchanged if no pattern is matched

from nltk.tokenize import word_tokenize
text = "Tokenization in a major technique in NLP. Word embeddings help models learn languages"
tokens = word_tokenize(text)
print(tokens)
# Output => ['Tokenization', 'in', 'a', 'major', 'technique', 'in', 'NLP', '.', 'Word', 'embeddings', 'help', 'models', 'learn', 'languages']

# Usage example
stemmer = RegexStemmer()
stems = [stemmer.stem(word) for word in tokens]
print(stems)
# Output => ['Tokenize', 'in', 'a', 'major', 'technique', 'in', 'NLP', '.', 'Word', 'embed', 'help', 'model', 'learn', 'language']
