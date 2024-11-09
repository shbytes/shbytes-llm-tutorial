import re    # Regular expression library

# Define text
text = "Tokenization in a major technique in NLP. Let's learn!"

# Define regex pattern for word tokenization
# r'\b\w+\b' - Regular expression - matches of any space or character between words
tokens = re.findall(r'\b\w+\b', text.lower())   # find all token and collect as list
print(tokens)
# Output: ['tokenization', 'in', 'a', 'major', 'technique', 'in', 'nlp', 'let', 's', 'learn']
