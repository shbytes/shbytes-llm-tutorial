# Using sentencepiece library
import sentencepiece as spm

# Train SentencePiece model
spm.SentencePieceTrainer.train(input='sentencepiece-training-data.txt', model_prefix='m', vocab_size=30)
# update vocab_size according to the training data

# Load trained model
sp = spm.SentencePieceProcessor(model_file='m.model')

# Tokenize text
tokens = sp.encode("Tokenization in a major technique in NLP.", out_type=str)
print("SentencePiece Tokens:", tokens)
# Output => SentencePiece Tokens: ['▁', 'T', 'o', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n', '▁i', 'n', '▁', 'a', '▁', 'm', 'a', 'j', 'o', 'r', '▁', 't', 'e', 'c', 'h', 'n', 'i', 'q', 'u', 'e', '▁i', 'n', '▁', 'N', 'L', 'P', '.']