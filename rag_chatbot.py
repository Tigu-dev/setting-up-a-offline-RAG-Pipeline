from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

#loading the document

print("Loading data.txt ...")
loader = TextLoader("data.txt", autodetect_encoding=True)
documents = loader.load()
print("Documents loaded:", len(documents))

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print("Chunks created:", len(chunks))

#Embeddings

print("Loading HuggingFace Embeddings...")
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#vectore Databases

print("Building FAISS vector database...")
db = FAISS.from_documents(chunks, embedding)

#local llm TinyLlama , good for low-end cpus and does'nt use gpu

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading local LLM...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)

#chat frontend


print("\n OFFLINE RAG Chatbot Started! Type 'exit' to quit.")

while True:
    query = input("\nYou: ")

    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    retrieved_docs = db.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in retrieved_docs])

    prompt = f"""
You are an offline RAG assistant.
Answer ONLY using this context.
If answer not found, reply: "Not found in your documents."

CONTEXT:
{context}

QUESTION: {query}
ANSWER:
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.4,
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.split("ANSWER:")[-1].strip()

    print("\n Answer:", answer)
