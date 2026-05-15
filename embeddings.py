import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Initialize embeddings globally to avoid reloading multiple times
_embeddings = None

def get_embeddings_model():
    """Returns the HuggingFace embeddings model, initializing it if necessary."""
    global _embeddings
    if _embeddings is None:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        _embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return _embeddings

def create_and_save_vectorstore(documents: list[Document], save_dir: str = "faiss_index") -> FAISS:
    """
    Creates a FAISS vectorstore from a list of documents and saves it to the specified directory.
    """
    try:
        embeddings_model = get_embeddings_model()
        vectorstore = FAISS.from_documents(documents, embeddings_model)
        
        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)
        vectorstore.save_local(save_dir)
        print(f"Vectorstore successfully saved to {save_dir}")
        return vectorstore
    except Exception as e:
        print(f"Error creating/saving vectorstore: {e}")
        return None

def load_vectorstore(load_dir: str = "faiss_index") -> FAISS:
    """
    Loads an existing FAISS vectorstore from the specified directory.
    """
    try:
        embeddings_model = get_embeddings_model()
        # allow_dangerous_deserialization=True is required for loading a local FAISS index
        vectorstore = FAISS.load_local(load_dir, embeddings_model, allow_dangerous_deserialization=True)
        return vectorstore
    except Exception as e:
        print(f"Error loading vectorstore from {load_dir}: {e}")
        return None
