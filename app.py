import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import requests

# Load API keys from Streamlit secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists, and create if not
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

# Connect to the index
index = pc.Index(INDEX_NAME)

# Load Sentence Transformer Model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Groq API setup
url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# Function to call Groq API
def get_groq_response(prompt):
    data = {
        "model": "deepseek-r1-distill-qwen-32b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Function to upload data to Pinecone
def upload_to_pinecone(df):
    vectors = []
    for idx, row in df.iterrows():
        question_id = str(idx)
        question = row["Question"]
        answer = row["Answer"]
        embedding = model.encode(question).tolist()
        vectors.append({
            "id": question_id,
            "values": embedding,
            "metadata": {"question": question, "answer": answer}
        })
    index.upsert(vectors=vectors, namespace="ns1")

# Streamlit App UI
st.title("📚 AI-Powered Question-Answer System")

# File uploader for Excel
uploaded_file = st.file_uploader("📂 Upload an Excel file (Columns: 'Question', 'Answer')", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("### ✅ Uploaded Data Preview:")
    st.write(df.head())

    if st.button("🚀 Upload Data to Pinecone"):
        with st.spinner("Uploading data to Pinecone..."):
            upload_to_pinecone(df)
        st.success("✅ Data successfully uploaded to Pinecone!")

# Query Section
query = st.text_input("🔍 Ask a question:")

if query:
    query_embedding = model.encode(query).tolist()
    results = index.query(namespace="ns1", vector=query_embedding, top_k=5, include_metadata=True)

    # Construct context for Groq API
    context = "\n".join([
        f"Question: {match['metadata']['question']}\nAnswer: {match['metadata']['answer']}\n"
        for match in results["matches"]
    ])

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = get_groq_response(prompt)

    st.write("### 🔍 Search Results from Pinecone:")
    for match in results["matches"]:
        st.write(f"**Q:** {match['metadata']['question']}")
        st.write(f"**A:** {match['metadata']['answer']}")
        st.write("---")

    st.write("### 🤖 AI Response:")
    st.write(response["choices"][0]["message"]["content"])
