import os
import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────
DOCS_DIR = "docs"
CHUNK_SIZE = 500      # words per chunk
CHUNK_OVERLAP = 50    # words of overlap between chunks
COLLECTION_NAME = "agilebuddha"

# ── Step 1: Chunking ───────────────────────────────────────
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks by word count"""
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
        
    return chunks

def load_and_chunk_docs(docs_dir):
    """Read all .txt files and split into chunks"""
    all_chunks = []
    
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith('.txt'):
            continue
            
        filepath = os.path.join(docs_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        # Extract metadata from the header we wrote earlier
        lines = full_text.split('\n')
        title = lines[0].replace('TITLE: ', '').strip()
        url = lines[1].replace('URL: ', '').strip()
        content = '\n'.join(lines[3:])  # skip header
        
        # Chunk the content
        chunks = chunk_text(content)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'text': chunk,
                'title': title,
                'url': url,
                'chunk_index': i,
                'filename': filename
            })
    
    print(f"Loaded {len(all_chunks)} chunks from {docs_dir}")
    return all_chunks

# ── Step 2: Embed and Store ────────────────────────────────
def store_in_chromadb(chunks):
    """Embed chunks and store in ChromaDB"""
    
    # Load the embedding model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Set up ChromaDB — persists to disk in ./chroma_store
    client = chromadb.PersistentClient(path="./chroma_store")
    
    # Delete collection if it exists (clean slate each run)
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    
    collection = client.create_collection(COLLECTION_NAME)
    
    # Process in batches of 50
    batch_size = 50
    total = len(chunks)
    
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        
        texts = [c['text'] for c in batch]
        ids = [f"chunk_{i+j}" for j, _ in enumerate(batch)]
        metadatas = [{
            'title': c['title'],
            'url': c['url'],
            'chunk_index': c['chunk_index'],
            'filename': c['filename']
        } for c in batch]
        
        # Generate embeddings
        embeddings = model.encode(texts).tolist()
        
        # Store in ChromaDB
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Stored chunks {i+1} to {min(i+batch_size, total)} of {total}")
    
    print(f"\nDone. {total} chunks stored in ChromaDB.")
    return collection

# ── Run it ─────────────────────────────────────────────────
print("Starting ingestion...\n")
chunks = load_and_chunk_docs(DOCS_DIR)
collection = store_in_chromadb(chunks)

# Quick verification
print(f"\nVerification — collection contains {collection.count()} chunks")
