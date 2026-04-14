import streamlit as st
from rag_pipeline import index_pdf, query_pdf
from openai import OpenAI
from dotenv import load_dotenv
import os
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
# 🔥 Load env
load_dotenv()

api_key = st.secrets["GEMINI_API_KEY"]
print("KEY LOADED:", api_key)

# 🚨 Safety check
if not api_key:
    st.error("API KEY NOT LOADED ❌")
    st.stop()

# 🔥 Create client (MISSING BEFORE)
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# UI
st.title("🤖 Chat with your PDF")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.session_state.vector_store = index_pdf("temp.pdf")
    st.success("PDF indexed!")

query = st.text_input("Ask something")

if query and st.session_state.vector_store:
    context = query_pdf(query, st.session_state.vector_store)

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "Answer ONLY from context"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )

    st.write(response.choices[0].message.content)