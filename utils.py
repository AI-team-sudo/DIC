import os
import PyPDF2
import hashlib
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec  # <-- Important update
 
# Load environment variables
load_dotenv()
 
# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # Optional if you're using it
pc = Pinecone(api_key=PINECONE_API_KEY)
 
# Define index name
index_name = "rfp-qna"
 
# Load embedding model
model = SentenceTransformer("thenlper/gte-large")
 
def hash_file(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()
 
def process_pdf_chunks(file_path, chunk_size=500):
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        full_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
 
def embed_chunks(chunks):
    return model.encode(chunks).tolist()
 
def ensure_index_exists():
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(  # <-- new required field
                cloud="aws",       # or "gcp"
                region="us-east-1" # change if needed
            )
        )
    return pc.Index(index_name)
 
def store_vectors_to_namespace(index, namespace, chunks, embeddings):
    vectors = [
        (f"{namespace}-chunk-{i}", embeddings[i], {"text": chunks[i]})
        for i in range(len(chunks))
    ]
    index.upsert(vectors=vectors, namespace=namespace)
 
def query_index(index, namespace, query):
    query_vector = model.encode([query])[0]
    results = index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True,
        namespace=namespace
    )
    return [match['metadata']['text'] for match in results.matches]
