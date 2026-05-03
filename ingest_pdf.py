import fitz  # pymupdf
import os
import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────
PDF_PATH = "book.pdf"          # rename to match your file
COLLECTION_NAME = "agilebuddha"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── Extract text from PDF ──────────────────────────────────
def extract_pdf_text(pdf_path):
    """Extract text page by page from PDF"""
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():  # skip blank pages
            pages.append({
                'text': text,
                'page': page_num + 1
            })
    
    print(f"Extracted {len(pages)} pages from PDF")
    return pages

# ── Chunk pages into pieces ────────────────────────────────
def chunk_pages(pages, pdf_path):
    """Combine all pages then chunk"""
    full_text = '\n'.join([p['text'] for p in pages])
    words = full_text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = ' '.join(words[start:end])
        chunks.append({
            'text': chunk,
            'title': 'Agile Estimation Distilled',
            'url': 'book',
            'source': os.path.basename(pdf_path),
            'chunk_index': len(chunks)
        })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    
    print(f"Created {len(chunks)} chunks from PDF")
    return chunks

# ── Add to existing ChromaDB collection ───────────────────
def add_to_chromadb(chunks):
    """Add PDF chunks to existing collection"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./chroma_store")
    collection = client.get_collection(COLLECTION_NAME)
    
    # Get current count to avoid ID conflicts
    current_count = collection.count()
    print(f"Existing chunks in collection: {current_count}")
    
    batch_size = 50
    total = len(chunks)
    
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        
        texts = [c['text'] for c in batch]
        ids = [f"pdf_chunk_{current_count + i + j}" 
               for j, _ in enumerate(batch)]
        metadatas = [{
            'title': c['title'],
            'url': c['url'],
            'source': c['source'],
            'chunk_index': c['chunk_index']
        } for c in batch]
        
        embeddings = model.encode(texts).tolist()
        
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Stored chunks {i+1} to {min(i+batch_size, total)} of {total}")
    
    print(f"\nDone. Collection now has {collection.count()} total chunks")

# ── Run it ─────────────────────────────────────────────────
print(f"Processing {PDF_PATH}...\n")
pages = extract_pdf_text(PDF_PATH)
chunks = chunk_pages(pages, PDF_PATH)
add_to_chromadb(chunks)
