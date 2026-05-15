from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def retrieve_relevant_chunks(vectorstore: FAISS, query: str, top_k: int = 4, distance_threshold: float = 1.2) -> list[Document]:
    """
    Retrieves the top_k relevant chunks for a given query from the vectorstore.
    Uses an L2 distance threshold to detect out-of-scope questions.
    (With FAISS L2 distance, a lower score indicates higher similarity).
    Returns a list of relevant Document objects, or an empty list if out of scope.
    """
    try:
        # returns list of (Document, score)
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=top_k)
        
        relevant_docs = []
        for doc, score in docs_and_scores:
            # lower L2 distance means higher similarity
            if score <= distance_threshold:
                relevant_docs.append(doc)
                
        # If no docs meet the threshold, we consider the query out of scope
        return relevant_docs
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []
