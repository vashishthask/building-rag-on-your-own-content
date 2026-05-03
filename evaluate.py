import os
import json
import datetime
import chromadb
from groq import Groq
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

# ── Ask ────────────────────────────────────────────────────
def ask(question):
    chunks = retrieve(question)
    prompt = build_prompt(question, chunks)

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )

    answer = response.choices[0].message.content

    # Score retrieval quality
    avg_score = sum(c['score'] for c in chunks) / len(chunks)
    if avg_score < 0.8:
        retrieval_quality = "STRONG"
    elif avg_score < 1.1:
        retrieval_quality = "MODERATE"
    else:
        retrieval_quality = "WEAK"

    return {
        'question': question,
        'answer': answer,
        'chunks': chunks,
        'retrieval_quality': retrieval_quality,
        'avg_score': round(avg_score, 4)
    }

# ── Evaluate ───────────────────────────────────────────────
def evaluate(test_cases):
    results = []
    os.makedirs("logs", exist_ok=True)

    print(f"\nRunning evaluation — {len(test_cases)} questions\n")
    print("=" * 60)

    for case in test_cases:
        question = case['question']
        category = case['category']

        result = ask(question)
        result['category'] = category

        # Print summary
        print(f"\nCategory: {category}")
        print(f"Question: {question}")
        print(f"Retrieval: {result['retrieval_quality']} (avg score: {result['avg_score']})")
        print(f"Top source: {result['chunks'][0]['title']}")
        print(f"Answer preview: {result['answer'][:200]}")
        print("-" * 60)

        results.append(result)

    # Save full results
    log = {
        'timestamp': datetime.datetime.now().isoformat(),
        'total_questions': len(results),
        'strong_retrieval': sum(1 for r in results if r['retrieval_quality'] == 'STRONG'),
        'moderate_retrieval': sum(1 for r in results if r['retrieval_quality'] == 'MODERATE'),
        'weak_retrieval': sum(1 for r in results if r['retrieval_quality'] == 'WEAK'),
        'results': results
    }

    with open('logs/evaluation.json', 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions:     {log['total_questions']}")
    print(f"Strong retrieval:    {log['strong_retrieval']}")
    print(f"Moderate retrieval:  {log['moderate_retrieval']}")
    print(f"Weak retrieval:      {log['weak_retrieval']}")
    print(f"\nFull results saved to logs/evaluation.json")

    return log

# ── Test cases ─────────────────────────────────────────────
# Four categories — each tests something different
test_cases = [

    # Category 1 — Your core expertise (should be STRONG)
    {"question": "What is a story point?",
     "category": "Core expertise"},
    {"question": "Should we even use story point? What if we dont? What if we map story point to hours.",
     "category": "Core expertise"},
    {"question": "When do you think specification by example may not make sense that much",
     "category": "Core expertise"},
    {"question": "When do you think we should use Kanban and when not",
     "category": "Core expertise"},
    {"question": "What's the problem in 3 questions based daily scrum",
     "category": "Core expertise"},
    {"question": "What's the problem in 3 questions based daily scrum",
     "category": "Core expertise"},
    

    # Category 2 — AI content (limited corpus — tests gap)
    {"question": "What happens to the job scenario because of AI",
     "category": "AI content"},
    

    # Category 3 — Edge of your corpus (tests weak retrieval)
    {"question": "Why do we estimate in the first place?",
     "category": "Corpus edge"},
    {"question": "What generally changes in Agile while moving from waterfall from Scrum Master perspective",
     "category": "Corpus edge"},
    

    # Category 4 — Outside corpus (should admit it doesn't know)
    {"question": "Where does coffee grow?",
     "category": "Outside corpus"},
]

evaluate(test_cases)
