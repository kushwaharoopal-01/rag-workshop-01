from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_documents(documents: list[Document], chunk_size: int = 700, chunk_overlap: int = 100) -> list[Document]:
    """
    Splits a list of LangChain Document objects into smaller chunks using RecursiveCharacterTextSplitter.
    Preserves existing metadata (filename, page number) in the resulting chunks.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        # split_documents automatically preserves metadata from the original Document objects
        chunks = text_splitter.split_documents(documents)
        return chunks
    except Exception as e:
        print(f"Error chunking documents: {e}")
        return []
