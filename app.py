import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import requests
import time  # Added for delay in batch uploads

# Load API Keys & Settings from Streamlit Secrets
try:
    PINECONE_API_KEY = st.secrets["pinecone"]["PINECONE_API_KEY"]
    PINECONE_INDEX = st.secrets["pinecone"]["PINECONE_INDEX"]
    PINECONE_HOST = st.secrets["pinecone"]["PINECONE_HOST"]  # Use if needed
    GROQ_API_KEY = st.secrets["groq"]["GROQ_API_KEY"]
except KeyError as e:
    st.error(f"Missing API Key in Streamlit secrets: {e}")
    st.stop()

# Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure the index exists
if PINECONE_INDEX not in pc.list_indexes().names():
    st.warning(f"Pinecone index '{PINECONE_INDEX}' not found. Creating a new one...")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the index
index = pc.Index(PINECONE_INDEX, host=PINECONE_HOST)

# Load Embedding Model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Groq API Setup
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# Function to Get Response from Groq API
def get_groq_response(prompt):
    data = {
        "model": "deepseek-r1-distill-qwen-32b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150
    }
    response = requests.post(GROQ_URL, headers=HEADERS, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Groq API Error: {response.text}")
        return None

# ✅ **Final Fixed Function to Upload Data to Pinecone in Batches**
def upload_to_pinecone(df, batch_size=100):
    if df.empty:
        st.error("Uploaded file is empty! Please upload a valid Excel file.")
        return
    
    vectors = []
    total_rows = df.shape[0]  # Count total rows
    st.write(f"**Total Rows in File: {total_rows}**")

    for idx, row in df.iterrows():
        question_id = str(idx)
        question = str(row["Question"]).strip()
        answer = str(row["Answer"]).strip()
        embedding = model.encode(question).tolist()

        vectors.append({
            "id": question_id,
            "values": embedding,
            "metadata": {"question": question, "answer": answer}
        })

        # ✅ **Upload in batches (e.g., 100 vectors per request)**
        if len(vectors) >= batch_size:
            try:
                index.upsert(vectors=vectors, namespace="ns1")
                st.write(f"Uploaded {len(vectors)} rows so far...")
                vectors = []  # Clear batch
                time.sleep(1)  # Small delay to prevent rate limits
            except Exception as e:
                st.error(f"Error uploading batch to Pinecone: {e}")
    
    # Upload remaining vectors if any
    if vectors:
        index.upsert(vectors=vectors, namespace="ns1")
        st.success(f"Final Upload Complete! Total {total_rows} rows uploaded to Pinecone.")

# Streamlit UI
st.title("Question-Answer AI (Powered by Pinecone & Groq)")

# Upload Excel File
uploaded_file = st.file_uploader("Upload an Excel file (must have 'Question' and 'Answer' columns)", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, engine="openpyxl")  # Force full data load
        st.write("### Uploaded Data Preview:")
        st.dataframe(df.head(10))  # Show first 10 rows
        st.write(f"Total Rows Loaded: **{df.shape[0]}**")  # Display actual row count
        
        # Upload Data to Pinecone
        if st.button("Upload Data to Pinecone"):
            with st.spinner("Uploading data to Pinecone..."):
                upload_to_pinecone(df)

    except Exception as e:
        st.error(f"Error reading Excel file: {e}")

# Query System
query = st.text_input("Enter your question:")

if query:
    query_embedding = model.encode(query).tolist()
    
    try:
        results = index.query(
            namespace="ns1",
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )

        if results and "matches" in results:
            context = ""
            for match in results["matches"]:
                context += f"Question: {match['metadata']['question']}\nAnswer: {match['metadata']['answer']}\n\n"

            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            response = get_groq_response(prompt)

            if response:
                st.write("### Refined LLM Response:")
                st.write(response['choices'][0]['message']['content'])
        else:
            st.warning("No matching results found in Pinecone.")

    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
