import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from utils import *
 
load_dotenv()
 
st.set_page_config(page_title="RFP Clarification Bot", layout="wide")
st.title("RFP Pre-Bid Clarification Bot")
 
uploaded_file = st.file_uploader("Upload your RFP PDF", type=["pdf"])
 
if uploaded_file:
    file_bytes = uploaded_file.read()
    file_hash = hash_file(file_bytes)
    namespace = f"rfp-{file_hash}"
 
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
 
    index = ensure_index_exists()
    index_desc = pc.describe_index(index_name)
 
    # Check if this namespace has already been stored
    existing_namespaces = index.describe_index_stats().namespaces.keys()
    if namespace in existing_namespaces:
        st.info("This RFP has already been uploaded and indexed.")
    else:
        st.write("Processing PDF...")
        chunks = process_pdf_chunks(tmp_path)
        embeddings = embed_chunks(chunks)
        store_vectors_to_namespace(index, namespace, chunks, embeddings)
        st.success("RFP uploaded and indexed successfully!")
 
    # Query UI
    st.subheader("Ask Pre-Bid Clarification")
    user_query = st.text_input("Enter your clarification query below:")
 
    if st.button("Get Clarification Response") and user_query:
        with st.spinner("Generating response..."):
            results = query_index(index, namespace, user_query)
            response = "\n\n".join(results) if results else "No relevant information found."
            st.markdown("Clarification Response")
            st.write(response)
