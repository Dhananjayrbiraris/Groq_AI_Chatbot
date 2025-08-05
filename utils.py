import os
import uuid
import chromadb    #type: ignore
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions       #type: ignore
from PyPDF2 import PdfReader
from groq import Groq  #type: ignore
from dotenv import load_dotenv

# Load env vars
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB persistent storage
chroma_client = chromadb.PersistentClient(path="db")
collection = chroma_client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
)

# In-memory chat history
chat_history = {}

# In-memory CSV storage {filename: DataFrame}
csv_tables = {}

def read_document(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        csv_tables[os.path.basename(file_path)] = df  # Save table for later
        text = df.to_string(index=False)
    return text.strip()

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def add_to_vector_db(file_path):
    text = read_document(file_path)
    chunks = chunk_text(text)
    if not chunks:
        return
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{os.path.basename(file_path)}_{uuid.uuid4().hex}" for _ in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)

def search_documents(query, top_k=4):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results["documents"][0] if results["documents"] else []

def is_table_question(question):
    """Detect if the question is about numeric/statistics for CSV."""
    keywords = ["average", "sum", "total", "max", "min", "count", "median", "mean"]
    return any(word in question.lower() for word in keywords)

def answer_csv_question(question):
    """Try to answer a question directly from stored CSV tables."""
    for filename, df in csv_tables.items():
        try:
            # Try detecting column name in question
            for col in df.columns:
                if col.lower() in question.lower():
                    # Numeric stats
                    if "average" in question.lower() or "mean" in question.lower():
                        return f"The average {col} is {df[col].mean()}"
                    elif "sum" in question.lower() or "total" in question.lower():
                        return f"The total {col} is {df[col].sum()}"
                    elif "max" in question.lower():
                        return f"The maximum {col} is {df[col].max()}"
                    elif "min" in question.lower():
                        return f"The minimum {col} is {df[col].min()}"
                    elif "count" in question.lower():
                        return f"The count of {col} entries is {df[col].count()}"
        except Exception:
            continue
    return None

def ask_groq(session_id, context, question):
    # CSV direct answer check
    if is_table_question(question):
        csv_answer = answer_csv_question(question)
        if csv_answer:
            return csv_answer

    # History
    history_text = ""
    if session_id in chat_history:
        for q, a in chat_history[session_id]:
            history_text += f"Q: {q}\nA: {a}\n"

    strict_prompt = f"""
You are a question-answering assistant. 
You are ONLY allowed to answer using the CONTEXT below. 
If the answer is not in the CONTEXT, reply with exactly: "I don't know".

CONTEXT:
{context}

CHAT HISTORY:
{history_text}

QUESTION:
{question}

Answer strictly based on the CONTEXT above.
"""

    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # High-quality Groq model
        messages=[{"role": "user", "content": strict_prompt}],
        temperature=0,  # Make deterministic
        max_tokens=512
    )
    answer = completion.choices[0].message.content

    if session_id not in chat_history:
        chat_history[session_id] = []
    chat_history[session_id].append((question, answer))

    return answer
