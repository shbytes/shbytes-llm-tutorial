# Hugging Face’s Tokenizers library
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())  # Initialize a BPE Tokenizer
trainer = trainers.BpeTrainer(vocab_size=1000, min_frequency=2) # create a trainer object

# Training data
training_text = ["Tokenization in a major technique in NLP. Let's learn it in detail!"]

# Train tokenizer
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.train_from_iterator(training_text, trainer)

# Tokenize input text
output = tokenizer.encode("Tokenization in a major technique in NLP.")
print("Subword Tokens:", output.tokens)
# Output => Subword Tokens: ['T', 'o', 'k', 'e', 'ni', 'z', 'a', 't', 'i', 'o', 'n', 'in', 'a', 'm', 'a', 'j', 'o', 'r', 't', 'e', 'c', 'h', 'ni', 'q', 'u', 'e', 'in', 'N', 'L', 'P', '.']
