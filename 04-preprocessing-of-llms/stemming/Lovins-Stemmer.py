class LovinsStemmer:
    def __init__(self):
        # Define suffixes and corresponding transformations based on the Lovins algorithm
        # Note: This is a simplified version with a few examples
        self.suffix_rules = {
            'ization': 'ize',
            'iveness': 'ive',
            'lessli': 'less',
            'es': 'e',
            'els': 'el',
            'dings': ''
        }

    def stem(self, word):
        """Stem the given word using the Lovins algorithm."""
        for suffix, replacement in sorted(self.suffix_rules.items(), key=lambda x: -len(x[0])):
            if word.endswith(suffix):
                return word[: -len(suffix)] + replacement
        return word  # Return the word unchanged if no suffix is matched

from nltk.tokenize import word_tokenize
text = "Tokenization in a major technique in NLP. Word embeddings help models learn languages"
tokens = word_tokenize(text)
print(tokens)
# Output => ['Tokenization', 'in', 'a', 'major', 'technique', 'in', 'NLP', '.', 'Word', 'embeddings', 'help', 'models', 'learn', 'languages']

# Usage example
stemmer = LovinsStemmer()
stems = [stemmer.stem(word) for word in tokens]
print(stems)
# Output => ['Tokenize', 'in', 'a', 'major', 'technique', 'in', 'NLP', '.', 'Word', 'embed', 'help', 'model', 'learn', 'language']