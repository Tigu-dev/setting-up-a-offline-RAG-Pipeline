import json
import os
import re

import fitz  # pip install pymupdf  (renders PDF pages, no poppler needed)
import pytesseract  # pip install pytesseract  (also install the Tesseract program, see README below)
import torch
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # pip install langchain-huggingface
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image, ImageOps  # pip install pillow
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------
# Config
# ----------------------------
DATA_FILE = "TAFL-UNIT-2-One-Shot-Notes.pdf"   # put the PDF in D:\RAG next to this script
OCR_CACHE_FILE = DATA_FILE + ".ocr.json"       # OCR runs once; results are cached here
INDEX_DIR = "faiss_index"

# Windows: path to the Tesseract program (default install location). Delete this line on Linux/Mac.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OCR_ZOOM = 2.0           # higher = sharper OCR but slower (2.0 is a good start)
MIN_PAGE_CHARS = 40      # pages with less text than this (cover, "Thank You", ads) are skipped
# Watermark / promo lines that repeat on every slide and would pollute the search
NOISE_PATTERN = re.compile(r"gateway|classes|7455|9612|download app|full courses", re.IGNORECASE)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 2                # small models get confused by too much context
MAX_DISTANCE = 1.1       # lower = stricter. If the best chunk is farther than this, the question
                         # is treated as irrelevant and the LLM is NOT called. Tune using DEBUG_SCORES.
DEBUG_SCORES = True      # prints the best distance for each question so you can tune MAX_DISTANCE
IRRELEVANT_MSG = "This question is not relevant to the documents I was given, so I can't answer it."
MAX_NEW_TOKENS = 200
SHOW_SOURCES = True      # print the retrieved chunks under each answer

SYSTEM_PROMPT = (
    "You are an offline RAG assistant. Answer ONLY using the provided context. "
    'If the answer is not in the context, reply exactly: "' + IRRELEVANT_MSG + '"'
)

# ----------------------------
# Embeddings
# ----------------------------
print("Loading embedding model...")
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)


# ----------------------------
# Vector DB (built once, then reused)
# ----------------------------
def clean_text(text):
    """Drop watermark/promo lines and blank lines."""
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln and not NOISE_PATTERN.search(ln))


def ocr_page(page):
    """Render one PDF page to an image and OCR it."""
    pix = page.get_pixmap(matrix=fitz.Matrix(OCR_ZOOM, OCR_ZOOM))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
    # These slides have a dark background: Tesseract works much better on dark text / light page
    if sum(img.resize((64, 36)).getdata()) / (64 * 36) < 100:
        img = ImageOps.invert(img)
    return pytesseract.image_to_string(img, config="--psm 6")


def load_pdf_pages():
    """Return {page_number: text}. Uses the PDF's own text if it has any, otherwise OCR (cached)."""
    if os.path.exists(OCR_CACHE_FILE) and os.path.getmtime(OCR_CACHE_FILE) > os.path.getmtime(DATA_FILE):
        print("Using cached OCR text...")
        with open(OCR_CACHE_FILE, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}

    pages = {}
    pdf = fitz.open(DATA_FILE)
    print(f"Reading {len(pdf)} pages (OCR is slow, this happens only once)...")
    for i, page in enumerate(pdf, start=1):
        text = page.get_text()
        if len(text.strip()) < MIN_PAGE_CHARS:  # no text layer -> scanned/image page -> OCR
            text = ocr_page(page)
        pages[i] = clean_text(text)
        print(f"  page {i}/{len(pdf)}", end="\r")
    print()

    with open(OCR_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=1)
    return pages


def build_index():
    print(f"Loading {DATA_FILE} ...")
    pages = load_pdf_pages()
    documents = [
        Document(page_content=text, metadata={"page": num, "source": DATA_FILE})
        for num, text in pages.items()
        if len(text) >= MIN_PAGE_CHARS
    ]
    print("Pages with usable text:", len(documents), "of", len(pages))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print("Chunks created:", len(chunks))

    print("Building FAISS vector database...")
    db = FAISS.from_documents(chunks, embedding)
    db.save_local(INDEX_DIR)
    return db


def index_is_fresh():
    index_file = os.path.join(INDEX_DIR, "index.faiss")
    return os.path.exists(index_file) and os.path.getmtime(index_file) > os.path.getmtime(DATA_FILE)


if index_is_fresh():
    print("Loading saved FAISS index...")
    db = FAISS.load_local(INDEX_DIR, embedding, allow_dangerous_deserialization=True)
else:
    db = build_index()

# ----------------------------
# Local LLM (TinyLlama, CPU only)
# ----------------------------
print("Loading local LLM...")
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    torch_dtype=torch.float32,
    device_map="cpu",
)
model.eval()


# ----------------------------
# Answering
# ----------------------------
def answer_question(query):
    results = db.similarity_search_with_score(query, k=TOP_K)

    best_distance = float(results[0][1])
    if DEBUG_SCORES:
        print(f"[debug] best distance: {best_distance:.3f} (cutoff {MAX_DISTANCE})")

    if MAX_DISTANCE is not None and best_distance > MAX_DISTANCE:
        return IRRELEVANT_MSG, []

    docs = [doc for doc, _ in results]
    context = "\n\n".join(d.page_content for d in docs)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy = deterministic, grounded answers
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (not the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer, docs


# ----------------------------
# Chat loop
# ----------------------------
print("\nOFFLINE RAG Chatbot Started! Type 'exit' to quit.")

while True:
    query = input("\nYou: ").strip()

    if not query:
        continue
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    answer, docs = answer_question(query)
    print("\nAnswer:", answer)

    if SHOW_SOURCES and docs:
        print("\nSources:")
        for i, d in enumerate(docs, 1):
            preview = d.page_content.replace("\n", " ")[:120]
            print(f"  [{i}] (slide {d.metadata.get('page', '?')}) {preview}...")