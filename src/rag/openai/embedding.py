import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from models import embedding_model
from config import logger


# ------------------------------------------- VARIABLES ------------------------------------------ #
SRC_DIR = Path(__file__).resolve().parent.parent.parent
CURRENT_DIR = Path(__file__).resolve()
PDF_PATH = SRC_DIR.parent / "docs" / "BG-039-02mar2026.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
KWARGS = 5


# ------------------------------------- EMBEDDING + CHUNKING ------------------------------------- #
def load_pdf() -> List[Document]:
    # Go up one level to the project root, then down into 'docs'
    pdf_path_str = str(PDF_PATH)

    if not os.path.exists(pdf_path_str):
        raise FileNotFoundError(f"Pdf not found in: {pdf_path_str}")

    # This load the pdf
    pdf_loader = PyPDFLoader(pdf_path_str)

    # Check if the pdf is there
    try:
        pages = pdf_loader.load()
        logger.info(" PDF has been loaded and has {len(pages)} pages")
    except Exception as e:
        logger.error("  PDF has been loaded and has {len(pages)} pages")
        raise

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    pages_split = text_splitter.split_documents(pages)

    return pages_split


# ---------------------------------------- VECTOR DATABASE --------------------------------------- #
def connection_database():
    # Persistence directory for the database
    db_directory = str(CURRENT_DIR.parent / "db")
    collection_name = "bg_data"

    print("\n==================== 🎲 DATABASE ====================\n")

    # If DB already exists → load it
    if os.path.exists(db_directory) and len(os.listdir(db_directory)) > 0:
        logger.info("Loading existing vector database...")

        vectorstore = Chroma(
            embedding_function=embedding_model,
            persist_directory=db_directory,
            collection_name=collection_name,
        )

    # If DB dont exists → create the db
    else:
        logger.info("Vector database not found. Creating embeddings...")

        # Create the directory if not exists
        if not os.path.exists(db_directory):
            logger.info("Created this directory: %s", db_directory)
            os.makedirs(db_directory)

        # Creating the chroma database using the embedding model
        pages_loaded = load_pdf()

        try:
            vectorstore = Chroma.from_documents(
                documents=pages_loaded,
                embedding=embedding_model,
                persist_directory=db_directory,
                collection_name=collection_name,
            )
        except Exception as error:
            print(f"Error at creating the database: {error}")
            raise

    # This is how we are going to retrieve the information from the vector database
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": KWARGS}
    )

    return retriever


if __name__ == "__main__":
    connection_database()
