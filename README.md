# Building RAG on Your Own Content

A practical RAG (Retrieval Augmented Generation) implementation 
built over 14 years of personal knowledge content from 
agilebuddha.in — designed to simulate the real challenges 
enterprises face when deploying internal AI knowledge systems.

## What is this repository?

This repository is a RAG implementation built over content 
generated across 14 years on agilebuddha.in. The idea is to 
simulate the challenges an enterprise would face when adding 
existing content to a RAG implementation — not just build 
something that works in a demo.

## Why I built it

- For someone starting their AI journey who wants a simple 
  but realistic RAG implementation — this is a practical 
  starting point.
- To understand and simulate the challenges enterprises face 
  when building similar implementations at scale.
- To share honest observations about what breaks, not just 
  what works.

## What I learned

**Access permissions are a hidden corpus problem**
111 public posts made it in. Content behind logins, internal 
wikis, and restricted SharePoints never reached the corpus. 
The most valuable organisational knowledge is often the most 
protected. RAG built on accessible content misses exactly 
what matters most.

**Volume is not depth**
207 chunks from 111 posts — less than 2 chunks per post on 
average. Organisations measure knowledge by document count. 
RAG exposes that quantity and depth are completely different 
things. Thousands of Confluence pages may represent genuine 
expertise in only a handful of areas.

**Model switching has integration cost**
Switching from Claude to Groq required changing the response 
parsing format. Same concept, different API contract. 
Enterprises assuming they can swap models freely will find 
integration rework at every layer. Model portability is an 
architecture decision, not just a procurement one.

## How to run it

**1. Set up the environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Set your Groq API key**
Create a free key at console.groq.com, then:
```bash
export GROQ_API_KEY=<your-groq-key>
```

**3. Fetch your content**
```bash
python scrape.py
```

**4. Chunk and index the content**
```bash
python ingest.py
```

**5. Test retrieval**
```bash
python retrieve.py
```

**6. Ask questions via terminal**
```bash
python ask.py
```

**7. Run the web interface**
```bash
streamlit run app.py --server.fileWatcherType none
```

## What's next

This is Project 1 of a three-part learning series built 
alongside a book on AI transformation in large enterprises. 
Project 2 will focus on AI-assisted delivery metrics and 
value stream analysis. Project 3 will explore whether AI 
helps skilled practitioners more than unskilled ones. The 
book, observations, and future projects will be linked here 
as they are completed.

## About

Built by ShriKant Vashishtha — Enterprise Agile Coach, 
Solution Architect, and AI Transformation Advisor. Writing 
a book on why AI transformation fails in large enterprises 
and what to do instead.
