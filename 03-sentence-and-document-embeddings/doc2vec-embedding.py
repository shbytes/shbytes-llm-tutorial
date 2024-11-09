from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Sample documents
documents = ["Sentence and document embeddings are fundamental to Natural Language Processing",
             "Sentence and document embeddings capture the meaning of text"]

# Prepare tagged documents
tagged_docs = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(documents)]

# Train Doc2Vec model
model = Doc2Vec(tagged_docs,
                vector_size=50, # Dimensions of the word vectors
                window=2,       # Maximum distance between the current and predicted word within a sentence
                min_count=1,    # Ignores all words with a total frequency lower than this
                workers=4)      # Use these many worker threads to train the model

# Obtain embeddings for a document
vector = model.infer_vector("Natural Language Processing is fascinating".split())
print("Doc2Vec Embedding:", vector)

# Output
# Doc2Vec Embedding: [ 0.00902946 -0.00518695 -0.00370304 -0.00914641  0.00105161  0.00624333
#  -0.00553964 -0.00713476  0.00331546 -0.00070394 -0.00828523 -0.00077945
#  -0.00827082 -0.00695372  0.00644662 -0.00731054  0.00989284 -0.00881199
#   0.00061673  0.00813323  0.00850085  0.00954109  0.00507595  0.00174437
#   0.0034364   0.003292    0.00108508 -0.0080248  -0.00848993  0.0077961
#   0.00923685  0.00371492  0.0002801  -0.00664406 -0.00984772 -0.00373417
#   0.00963134 -0.00376201 -0.00385672  0.00354381 -0.00094189 -0.0012201
#   0.00931827 -0.00975919 -0.00565285  0.00674632  0.00764528  0.005375
#   0.0045256  -0.00128563]