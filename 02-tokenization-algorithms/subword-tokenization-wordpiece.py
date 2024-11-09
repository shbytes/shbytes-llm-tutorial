# wordpiece lagorithm using Hugging Face’s Tokenizer
from transformers import BertTokenizer

# Load pre-trained WordPiece tokenizer model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input text
text = "Tokenization in a major technique in NLP."
tokens = tokenizer.tokenize(text)   # Tokenize the given text using wordpiece algorithm
print("WordPiece Tokens:", tokens)
# Output => WordPiece Tokens: ['token', '##ization', 'in', 'a', 'major', 'technique', 'in', 'nl', '##p', '.']