from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load a pre-trained Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# List of documents to search
documents = [
    "Machine learning models are revolutionizing healthcare.",
    "Natural language processing enables machines to understand text.",
    "Data science involves extracting insights from data.",
    "Sentence and document embeddings are fundamental to Natural Language Processing",
    "Sentence and document embeddings capture the meaning of text"
]

# Example query
query = "How sentence embedding is more effective?"

# Generate embeddings for documents
document_embeddings = model.encode(documents, convert_to_tensor=True)

# Generate embedding for the query
query_embedding = model.encode(query, convert_to_tensor=True)

# Compute cosine similarities
cosine_scores = util.cos_sim(query_embedding, document_embeddings)

# Find the top N most similar documents
top_k = 3  # Number of top documents to retrieve
top_results = np.argsort(-cosine_scores[0])[:top_k]

# Display the top results
print("Query:", query)
print("\nTop Documents:")

for idx in top_results:
    print(f"Document: {documents[idx]}")
    print(f"Score: {cosine_scores[0][idx].item():.4f}")
    print("-" * 50)

# Output
# Query: How sentence embedding is more effective?
#
# Top Documents:
# Document: Sentence and document embeddings are fundamental to Natural Language Processing
# Score: 0.6793
# --------------------------------------------------
# Document: Sentence and document embeddings capture the meaning of text
# Score: 0.6661
# --------------------------------------------------
# Document: Natural language processing enables machines to understand text.
# Score: 0.3207
# --------------------------------------------------