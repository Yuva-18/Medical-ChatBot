import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain_pinecone import PineconeEmbeddings

def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Given a list of documents objects, return a new list of Document
    objects containing only 'source' in metadata and the original page_content."""

    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts_chunks = text_splitter.split_documents(minimal_docs)
    return texts_chunks

def get_embeddings():
    """Return Pinecone's hosted embedding model (API-based, no local model
    download). Uses the Pinecone account's own inference API (5M free
    tokens/month) rather than Gemini's much smaller free embedding quota
    (1000 requests/day), which is too small to re-embed this project's PDF
    in a reasonable time."""
    return PineconeEmbeddings(
        model="multilingual-e5-large",
        pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
    )
