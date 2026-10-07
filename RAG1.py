from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ----------------------------
# 1. Documents (Knowledge Base)
# ----------------------------
documents = [
    "Python is a popular programming language used for AI, ML, and automation.",
    "LangChain is a framework for developing applications powered by large language models.",
    "RAG stands for Retrieval-Augmented Generation. It retrieves external knowledge before answering.",
    "FAISS is a vector search library created by Facebook for efficient similarity search."
]

# ----------------------------
# 2. Load Embedding Model
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert docs to embeddings
doc_embeddings = model.encode(documents)

# Convert to numpy float32 (FAISS requirement)
doc_embeddings = np.array(doc_embeddings).astype("float32")

# ----------------------------
# 3. Create FAISS Index
# ----------------------------
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# ----------------------------
# 4. Query
# ----------------------------
query = "What is RAG in AI?"
query_embedding = model.encode([query]).astype("float32")

# Search for top-1 most similar document
distance, doc_index = index.search(query_embedding, k=1)

retrieved_doc = documents[doc_index[0][0]]

print("Query:", query)
print("\nRetrieved Document:", retrieved_doc)

# ----------------------------
# 5. Simulated LLM Response
# ----------------------------
final_answer = f"""
RAG (Retrieval-Augmented Generation) is an AI technique where a model retrieves 
relevant external information before generating an answer. 

Based on your query, here is the retrieved context:
"{retrieved_doc}"

Using this context, the model can produce a more accurate and factual answer.
"""

print("\nFinal RAG Answer:")
print(final_answer)
