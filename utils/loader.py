import fitz  # PyMuPDF
from langchain_core.documents import Document
import os

def load_pdf(file_path: str) -> list[Document]:
    """
    Extracts text from a PDF file using PyMuPDF and returns a list of LangChain Document objects.
    Preserves metadata such as filename and page number.
    """
    documents = []
    filename = os.path.basename(file_path)
    
    try:
        # Open the PDF file
        pdf_document = fitz.open(file_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            
            # Create a Document for each page if text is not empty
            if text.strip():
                metadata = {
                    "filename": filename,
                    "page_number": page_num + 1  # 1-indexed page number
                }
                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)
                
        pdf_document.close()
        return documents
    except Exception as e:
        print(f"Error loading PDF {file_path}: {e}")
        return []
