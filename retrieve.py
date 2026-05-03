import chromadb
from sentence_transformers import SentenceTransformer

# ── Setup ──────────────────────────────────────────────────
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_collection("agilebuddha")

# ── Retrieval ──────────────────────────────────────────────
def retrieve(question, n_results=3):
    """Find the most relevant chunks for a question"""
    
    # Convert question to embedding
    question_embedding = model.encode(question).tolist()
    
    # Search ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )
    
    chunks = []
    for i in range(len(results['documents'][0])):
        chunks.append({
            'text': results['documents'][0][i],
            'title': results['metadatas'][0][i]['title'],
            'url': results['metadatas'][0][i]['url'],
            'score': results['distances'][0][i]
        })
    
    return chunks

# ── Test it ────────────────────────────────────────────────
def test_retrieval(question):
    print(f"\nQuestion: {question}")
    print("-" * 60)
    
    chunks = retrieve(question)
    
    for i, chunk in enumerate(chunks):
        print(f"\nResult {i+1}:")
        print(f"Title: {chunk['title']}")
        print(f"URL: {chunk['url']}")
        print(f"Relevance score: {chunk['score']:.4f}")
        print(f"Content preview: {chunk['text'][:200]}")
        print()

# ── Run three test questions ───────────────────────────────
test_retrieval("What do you think about WIP limits?")
test_retrieval("How should teams approach estimation?")
test_retrieval("What is the relationship between AI and team productivity?")
