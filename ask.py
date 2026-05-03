import os
import json
import datetime
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

# ── Setup ──────────────────────────────────────────────────
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_collection("agilebuddha")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Retrieval ──────────────────────────────────────────────
def retrieve(question, n_results=3):
    question_embedding = model.encode(question).tolist()
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

# ── Build prompt ───────────────────────────────────────────
def build_prompt(question, chunks):
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n--- Source {i+1}: {chunk['title']} ---\n"
        context += chunk['text']
        context += "\n"
    
    return f"""You are an AI assistant for ShriKant Vashishtha,
an enterprise agile coach and AI transformation advisor.

Answer the question using ONLY the sources provided below.
If the sources don't contain enough information to answer
the question, say clearly: "I don't have enough information
in the available content to answer this fully."

When answering, naturally reference which source you drew 
from — for example: "In your post on WIP limits, you argue..."
or "Your book Agile Estimation Distilled explains..."

Do not make up information. Do not use general knowledge.
Only use what is in the sources.

SOURCES:
{context}

QUESTION: {question}

ANSWER:"""

# ── Log to file ────────────────────────────────────────────
def log_interaction(question, answer, chunks):
    os.makedirs("logs", exist_ok=True)
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "sources": [
            {
                "title": c['title'],
                "url": c['url'],
                "relevance_score": round(c['score'], 4)
            }
            for c in chunks
        ]
    }
    
    with open("logs/interactions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

# ── Ask ────────────────────────────────────────────────────
def ask(question):
    print(f"\nQuestion: {question}")
    print("-" * 60)
    
    chunks = retrieve(question)
    prompt = build_prompt(question, chunks)
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    
    answer = response.choices[0].message.content
    
    # Log it
    log_interaction(question, answer, chunks)
    
    # Print answer
    print(f"\nAnswer:\n{answer}")
    
    # Print sources
    print(f"\nSources:")
    for i, chunk in enumerate(chunks):
        score_label = "strong" if chunk['score'] < 0.8 else \
                      "moderate" if chunk['score'] < 1.1 else "weak"
        print(f"  {i+1}. [{score_label}] {chunk['title']}")
        print(f"     {chunk['url']}")
    
    reply = {"answer": answer,
        "sources": [
            {
                "title": c['title'],
                "url": c['url'],
                "relevance_score": round(c['score'], 4)
            }
            for c in chunks
        ]}
    
    return reply

# ── 10 test questions ──────────────────────────────────────
if __name__ == "__main__":
    questions = [
        # Things you've written about
        "What is the relationship between AI and team productivity?",
        "How should teams approach estimation?",
        "What do you think about WIP limits?",
        "What makes a good Agile transformation?",
        "How do you run an effective daily standup?",
        
        # Things that test the edges
        "What is your view on SAFe?",
        "How should a CIO approach AI governance?",
        "What do you think about microservices architecture?",
        "How do you measure the success of an Agile transformation?",
        
        # Something clearly outside your corpus
        "What is the best way to cook pasta?"
    ]

    for q in questions:
        ask(q)
        print("\n" + "="*60)
