import streamlit as st
import pandas as pd
import requests
import time
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Load API Keys & Settings from Streamlit Secrets
try:
    PINECONE_API_KEY = st.secrets["pinecone"]["PINECONE_API_KEY"]
    PINECONE_INDEX = st.secrets["pinecone"]["PINECONE_INDEX"]
    PINECONE_HOST = st.secrets["pinecone"]["PINECONE_HOST"]  
    GROQ_API_KEY = st.secrets["groq"]["GROQ_API_KEY"]
except KeyError as e:
    st.error(f"Missing API Key in Streamlit secrets: {e}")
    st.stop()

# GitHub Raw File URL
GITHUB_RAW_URL = "https://raw.githubusercontent.com/mhd-faizzan/viadeepseek/main/data.xlsx"

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

# Function to Fetch Data from GitHub
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_data_from_github():
    try:
        response = requests.get(GITHUB_RAW_URL)
        response.raise_for_status()
        df = pd.read_excel(response.content, engine="openpyxl")
        return df
    except Exception as e:
        st.error(f"Error fetching data from GitHub: {e}")
        return None

# Function to Upload Data to Pinecone
def upload_to_pinecone(df, batch_size=50):
    if df.empty:
        st.error("Fetched data is empty!")
        return

    vectors = []
    total_rows = df.shape[0]
    progress_bar = st.progress(0)

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

        if len(vectors) >= batch_size:
            try:
                index.upsert(vectors=vectors, namespace="ns1")
                vectors = []
                time.sleep(1)
            except Exception as e:
                st.error(f"Error uploading batch to Pinecone: {e}")

        progress_bar.progress((idx + 1) / total_rows)

    if vectors:
        index.upsert(vectors=vectors, namespace="ns1")
        st.success(f"Data upload complete! {total_rows} rows uploaded.")

# Fetch and Upload Data at App Startup
st.write("Fetching latest data from GitHub...")
df = fetch_data_from_github()
if df is not None:
    upload_to_pinecone(df)

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

            st.write("### Answer:")
            st.write(context)
        else:
            st.warning("No matching results found in Pinecone.")

    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
