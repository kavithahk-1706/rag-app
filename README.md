# Chat with your PDF — RAG Web App

A Retrieval-Augmented Generation (RAG) app: upload a PDF, ask questions, and get answers grounded only in that document. Works on mobile browsers.

## How it works
1. **Load** – text is extracted from the uploaded PDF (`pypdf`)
2. **Chunk** – text is split into 800-character chunks with 150-character overlap
3. **Embed** – each chunk becomes a vector using Gemini `gemini-embedding-001`
4. **Retrieve** – the question is embedded and the top 4 chunks are found by cosine similarity (NumPy)
5. **Generate** – the retrieved chunks + question are sent to `gemini-2.5-flash`, which answers only from that context

## Tech stack
Python, Streamlit, Google Gemini API, NumPy, pypdf

## Run locally
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
streamlit run app.py
```

## Deploy
Deployed on Streamlit Community Cloud with `GEMINI_API_KEY` set in the app's Secrets.
