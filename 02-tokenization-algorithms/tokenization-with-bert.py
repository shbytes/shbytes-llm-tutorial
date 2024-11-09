from transformers import BertTokenizer

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Sample text
text = "Tokenization in a major technique in NLP."

# Preprocessing and Tokenization
# Tokenize text with padding and truncation
encoded_input = tokenizer(text, padding=True, truncation=True, max_length=10, return_tensors="pt")

# Display tokenized output
print("Token IDs:", encoded_input['input_ids'])
print("Token Type IDs:", encoded_input['token_type_ids'])
print("Attention Mask:", encoded_input['attention_mask'])
# Output
# Token IDs: tensor([[  101, 19204,  3989,  1999,  1037,  2350,  6028,  1999, 17953,   102]])
# Token Type IDs: tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# Attention Mask: tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
