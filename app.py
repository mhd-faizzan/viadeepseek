import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import requests

# Step 1: Load API Keys & Settings from Streamlit Secrets
PINECONE_API_KEY = st.secrets["pinecone"]["PINECONE_API_KEY"]
PINECONE_INDEX = st.secrets["pinecone"]["PINECONE_INDEX"]
GROQ_API_KEY = st.secrets["groq"]["GROQ_API_KEY"]

# Step 2: Initialize Pinecone Client with Hosted Index
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=384,  # Match embedding model output
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the index
index = pc.Index(PINECONE_INDEX)

# Step 3: Load Embedding Model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Step 4: Groq API Setup
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def get_groq_response(prompt):
    """ Calls Groq API for LLM response. """
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop": ["According to", "Based on", "As per the information"]
    }
    response = requests.post(GROQ_URL, headers=HEADERS, json=data)
    result = response.json()
    
    llm_response = result['choices'][0]['message']['content']

    # Remove unwanted AI disclaimers
    remove_phrases = ["According to the information provided,", "Based on the given data,", "As per the details you provided,"]
    for phrase in remove_phrases:
        llm_response = llm_response.replace(phrase, "").strip()

    return llm_response

# Step 5: Function to Upload Data to Pinecone
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
            "metadata": {
                "question": question,
                "answer": answer
            }
        })
    
    try:
        index.upsert(vectors=vectors, namespace="ns1")
        st.success("All data uploaded to Pinecone successfully!")
    except Exception as e:
        st.error(f"Error uploading data to Pinecone: {e}")

# Step 6: Streamlit UI
st.title("Question-Answer AI (Powered by Pinecone & Groq)")

# Step 7: Upload Excel File
uploaded_file = st.file_uploader("Upload an Excel file (must have 'Question' and 'Answer' columns)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    st.write("### Uploaded Data Preview:")
    st.write(df.head())

    if st.button("Upload Data to Pinecone"):
        with st.spinner("Uploading data to Pinecone..."):
            upload_to_pinecone(df)

# Step 8: Query System
query = st.text_input("Enter your question:")

if query:
    query_embedding = model.encode(query).tolist()
    results = index.query(namespace="ns1", vector=query_embedding, top_k=5, include_metadata=True)

    # Step 1: Extract answers from Pinecone
    context = ""
    for match in results["matches"]:
        if match['score'] >= 0.7:  # Ensures only relevant answers are used
            context += f"Question: {match['metadata']['question']}\nAnswer: {match['metadata']['answer']}\n\n"

    # Step 2: If context is found, use it; otherwise, ask the LLM directly
    if context.strip():
        prompt = f"Use the following information to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    else:
        prompt = f"Answer this question using your knowledge:\n\n{query}"

    response = get_groq_response(prompt)

    # Step 3: Display the final response
    st.write("### AI Response:")
    st.write(response)
