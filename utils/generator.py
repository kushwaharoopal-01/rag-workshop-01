import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

def generate_answer(query: str, retrieved_docs: list[Document]) -> dict:
    """
    Generates an answer using Google Gemini API based on the retrieved context.
    Prevents hallucinations by using a grounded RAG prompt.
    Returns the answer and the citations used.
    """
    # Detect out-of-scope questions by checking if docs are empty
    if not retrieved_docs:
        return {
            "answer": "I'm sorry, I don't have enough context in the uploaded documents to answer that question. It appears to be out of scope based on the provided material.",
            "citations": []
        }
    
    # Format the context and extract unique citations
    context = ""
    citations = []
    
    for i, doc in enumerate(retrieved_docs):
        filename = doc.metadata.get("filename", "Unknown file")
        page_num = doc.metadata.get("page_number", "Unknown page")
        citation_ref = f"[{i+1}] {filename} (Page {page_num})"
        
        context += f"Document {citation_ref}:\n{doc.page_content}\n\n"
        
        if citation_ref not in citations:
            citations.append(citation_ref)
            
    # Define the grounded prompt to prevent hallucinations
    prompt_template = """
    You are an academic assistant known as "Study Buddy". 
    Your task is to answer the user's question ONLY using the provided context below. 
    If the context does not contain the answer, you must state that you don't know and do not attempt to guess or hallucinate.
    When answering, cite your sources using the document references provided (e.g., [1], [2]).
    
    Context:
    {context}
    
    User Question:
    {question}
    
    Answer:
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    try:
        # Initialize Gemini 1.5 Flash (Requires GOOGLE_API_KEY environment variable)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.0,  # 0 temperature to reduce hallucinations
            max_output_tokens=1024,
        )
        
        chain = prompt | llm
        
        response = chain.invoke({
            "context": context,
            "question": query
        })
        
        return {
            "answer": response.content,
            "citations": citations
        }
    except Exception as e:
        print(f"Error generating answer: {e}")
        return {
            "answer": f"An error occurred while communicating with the Gemini API: {str(e)}",
            "citations": []
        }
