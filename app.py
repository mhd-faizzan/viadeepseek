import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import pandas as pd
import requests

# Step 3: Load API Keys & Settings from Streamlit Secrets
PINECONE_API_KEY = st.secrets["pinecone"]["PINECONE_API_KEY"]
PINECONE_HOST = st.secrets["pinecone"]["PINECONE_HOST"]
PINECONE_INDEX = st.secrets["pinecone"]["PINECONE_INDEX"]
GROQ_API_KEY = st.secrets["groq"]["GROQ_API_KEY"]

# Step 4: Initialize Pinecone Client with Hosted Index
pc = Pinecone(api_key=PINECONE_API_KEY, host=PINECONE_HOST)
index = pc.Index(PINECONE_INDEX)

# Step 5: Load Embedding Model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Step 6: Groq API Setup
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def get_groq_response(prompt):
    data = {
        "model": "deepseek-r1-distill-qwen-32b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150
    }
    response = requests.post(GROQ_URL, headers=HEADERS, json=data)
    return response.json()

# Step 7: Function to Upload Data to Pinecone
def upload_to_pinecone(df):
    vectors = []
    for idx, row in df.iterrows():
        question_id = str(idx)
        question = row["Question"]
        answer = row["Answer"]
        embedding = model.encode(question)  # Embed the question
        vectors.append({
            "id": question_id,
            "values": embedding.tolist(),  # Convert numpy array to list
            "metadata": {
                "question": question,
                "answer": answer
            }
        })
    index.upsert(vectors=vectors, namespace="ns1")  # Use namespace "ns1"

# Step 8: Streamlit App UI
st.title("Question-Answer AI (Powered by Pinecone & Groq)")

# Step 9: Upload Excel File
uploaded_file = st.file_uploader("Upload an Excel file (must have 'Question' and 'Answer' columns)", type=["xlsx"])

if uploaded_file is not None:
    # Read the uploaded Excel file
    df = pd.read_excel(uploaded_file, engine="openpyxl")  # Ensure correct engine
    st.write("### Uploaded Data Preview:")
    st.write(df.head())

    # Upload data to Pinecone
    if st.button("Upload Data to Pinecone"):
        with st.spinner("Uploading data to Pinecone..."):
            upload_to_pinecone(df)
        st.success("Data uploaded to Pinecone successfully!")

# Step 10: Query System
query = st.text_input("Enter your question:")

if query:
    # Step 1: Search Pinecone
    query_embedding = model.encode(query).tolist()  # Convert numpy array to list
    results = index.query(
        namespace="ns1",
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    # Step 2: Use Groq API for response
    context = ""
    for match in results["matches"]:
        context += f"Question: {match['metadata']['question']}\nAnswer: {match['metadata']['answer']}\n\n"

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = get_groq_response(prompt)

    # Display results
    st.write("### Search Results from Pinecone:")
    for match in results["matches"]:
        st.write(f"**Question:** {match['metadata']['question']}")
        st.write(f"**Answer:** {match['metadata']['answer']}")
        st.write("---")

    st.write("### LLM Response:")
    st.write(response['choices'][0]['message']['content'])
