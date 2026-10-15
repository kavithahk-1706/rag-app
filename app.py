"""
Simple RAG (Retrieval-Augmented Generation) web app.

Flow:
  1. User uploads a PDF
  2. We extract the text and split it into overlapping chunks
  3. Each chunk is turned into an embedding (a vector of numbers) using Gemini
  4. When the user asks a question, we embed the question too
  5. We find the chunks whose vectors are most similar to the question (cosine similarity)
  6. We send those chunks + the question to Gemini, which answers using only that context
"""

import os
import time
import numpy as np
import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import errors

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks
TOP_K = 6             # how many chunks to retrieve per question


# ---------- step 1: read the pdf ----------
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            # some pdfs extract one word per line; this collapses all extra
            # spaces/newlines into single spaces so chunks hold real content
            pages.append(" ".join(text.split()))
    return "\n".join(pages)


# ---------- step 2: split into chunks ----------
def split_into_chunks(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    # overlap means a sentence cut at the edge of one chunk still appears whole in the next
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


# ---------- step 3 & 4: embeddings ----------
def embed_batch(client, batch, task_type):
    # the free tier has a rate limit, so if we get error 429 ("too many requests")
    # we wait a bit and try again instead of crashing
    for attempt in range(5):
        try:
            return client.models.embed_content(
                model=EMBED_MODEL,
                contents=batch,
                config={"task_type": task_type},
            )
        except errors.ClientError as e:
            if e.code == 429 and attempt < 4:
                time.sleep(15)
            else:
                raise


def embed_texts(client, texts, task_type):
    vectors = []
    batch_size = 20  # small batches so we don't hit the free-tier limit too fast
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = embed_batch(client, batch, task_type)
        for item in result.embeddings:
            vectors.append(item.values)
    return np.array(vectors)


# ---------- step 5: retrieval ----------
def find_relevant_chunks(question_vector, chunk_vectors, chunks, k=TOP_K):
    # cosine similarity = dot product of the two vectors divided by their lengths
    dot = chunk_vectors @ question_vector
    lengths = np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(question_vector)
    scores = dot / lengths
    best_indexes = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in best_indexes]


# ---------- step 6: generation ----------
def answer_question(client, question, relevant_chunks):
    context = "\n\n---\n\n".join(chunk for chunk, _ in relevant_chunks)
    prompt = f"""You are a helpful assistant answering questions about a document.
Answer the question using ONLY the context below.
Give a complete answer in full sentences: include the relevant titles, names,
aims or details from the context, not just a label or heading.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}
"""
    response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return response.text


# ---------- the web page ----------
st.set_page_config(page_title="PDF Q&A (RAG)", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption("Upload a PDF, then ask questions about it. Answers come only from the document.")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None
if not api_key:
    st.error("GEMINI_API_KEY is not set. Add it in the app's secrets.")
    st.stop()
client = genai.Client(api_key=api_key)

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    # only re-process when a new file is uploaded (streamlit reruns the script on every click)
    if st.session_state.get("file_name") != uploaded.name:
        with st.spinner("Reading and indexing the PDF..."):
            text = extract_text(uploaded)
            if not text.strip():
                st.error("Couldn't extract any text. Is this a scanned image PDF?")
                st.stop()
            chunks = split_into_chunks(text)
            try:
                vectors = embed_texts(client, chunks, "RETRIEVAL_DOCUMENT")
            except errors.APIError as e:
                # show the real reason instead of streamlit's redacted error
                st.error(f"Gemini API error {e.code}: {e.message}")
                st.stop()
            st.session_state.chunks = chunks
            st.session_state.vectors = vectors
            st.session_state.file_name = uploaded.name
            st.session_state.history = []
        st.success(f"Indexed {len(st.session_state.chunks)} chunks from {uploaded.name}")

    # show previous messages
    for role, message in st.session_state.history:
        st.chat_message(role).write(message)

    question = st.chat_input("Ask something about the PDF")
    if question:
        st.chat_message("user").write(question)
        with st.spinner("Thinking..."):
            try:
                q_vector = embed_texts(client, [question], "RETRIEVAL_QUERY")[0]
                relevant = find_relevant_chunks(q_vector, st.session_state.vectors, st.session_state.chunks)
                answer = answer_question(client, question, relevant)
            except errors.APIError as e:
                st.error(f"Gemini API error {e.code}: {e.message}")
                st.stop()
        st.chat_message("assistant").write(answer)
        with st.expander("Retrieved chunks (the 'R' in RAG)"):
            for chunk, score in relevant:
                st.markdown(f"**similarity: {score:.3f}**")
                st.text(chunk[:500])
        st.session_state.history.append(("user", question))
        st.session_state.history.append(("assistant", answer))
else:
    st.info("Upload a PDF to get started.")