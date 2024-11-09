from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained Sentence-BERT model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Example sentence
sentence = "Sentence and document embeddings are fundamental to Natural Language Processing"

# Tokenize and encode the sentence
inputs = tokenizer(sentence, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Extract embeddings from the last hidden state
embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

print("Embedding Shape:", embedding.shape)
print("Embedding:", embedding)

# Output
# Embedding Shape: torch.Size([384])
# Embedding: tensor([ 1.4393e-01, -7.0336e-02,  4.5347e-01,  7.5642e-02,  2.4421e-01,
#          3.2003e-01,  4.4435e-02,  9.9099e-02,  2.9994e-01, -2.8900e-01,
#          7.2729e-02,  1.4307e-01,  3.7878e-01,  1.1841e-01, -7.8208e-02,
#          2.2986e-01,  2.0121e-01,  1.7003e-01, -5.3624e-01, -2.0893e-01,
# ......
# ......
#         -4.0043e-02, -4.0156e-01,  4.8177e-02,  2.2681e-01, -1.1275e-01,
#          1.1733e-01, -6.3902e-02, -1.8911e-01, -1.1112e-01, -1.1332e-02,
#         -2.4947e-02,  5.0459e-01,  4.4430e-01,  1.8361e-02])