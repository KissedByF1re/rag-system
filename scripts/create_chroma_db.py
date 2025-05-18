#!/usr/bin/env python
"""
Script to create a Chroma database from the Russian RAG test dataset.
"""

import os
import sys
import pickle
import logging
import re
from typing import List, Dict, Any, Optional

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_ru_rag_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Load the Russian RAG test dataset pickle file.
    
    Args:
        dataset_path: Path to the pickle file
        
    Returns:
        The loaded dataset
    """
    logger.info(f"Loading dataset from {dataset_path}")
    try:
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def process_text_files(folder_path: str, chunk_size: int = 800, chunk_overlap: int = 400) -> List[Dict[str, str]]:
    """
    Process text files in a folder, splitting them into chunks with better context.
    
    Args:
        folder_path: Path to the folder containing text files
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks with metadata
    """
    logger.info(f"Processing text files from {folder_path} with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    # Use better separators for Russian text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", "],
        length_function=len,
        is_separator_regex=False,
    )
    
    documents = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            file_id = filename.split('.')[0]
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Detect document structure
                has_headers = bool(re.search(r'^#{1,3}\s+.+$', content, re.MULTILINE))
                
                # Split the text into chunks
                chunks = text_splitter.create_documents(
                    [content], 
                    metadatas=[{"source": file_path, "file_id": file_id}]
                )
                
                # Convert LangChain documents to dictionaries with enhanced metadata
                for i, chunk in enumerate(chunks):
                    # Add surrounding context
                    prev_context = ""
                    next_context = ""
                    
                    if i > 0:
                        prev_context = chunks[i-1].page_content[-200:]  # Last 200 chars
                    if i < len(chunks) - 1:
                        next_context = chunks[i+1].page_content[:200]  # First 200 chars
                    
                    documents.append({
                        "page_content": chunk.page_content,
                        "metadata": {
                            **chunk.metadata,
                            "chunk_id": i,
                            "total_chunks": len(chunks),
                            "has_structure": has_headers,
                            "prev_context": prev_context,
                            "next_context": next_context,
                            "content_length": len(chunk.page_content)
                        }
                    })
                
                logger.info(f"Processed {filename}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
    
    logger.info(f"Total chunks created: {len(documents)}")
    return documents

def create_chroma_db(documents: List[Dict[str, str]], persist_directory: str, collection_name: str = "ru_rag_collection") -> chromadb.Collection:
    """
    Create a Chroma database from document chunks.
    
    Args:
        documents: List of document chunks with metadata
        persist_directory: Directory to persist the database
        collection_name: Name of the collection
        
    Returns:
        The Chroma collection
    """
    logger.info(f"Creating Chroma database in {persist_directory} with collection {collection_name}")
    
    # Create the persist directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    
    # Initialize the embedding function using Hugging Face
    logger.info("Initializing HuggingFace embeddings (deepvk/USER-base)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="deepvk/USER-base",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    logger.info("Embeddings initialized")
    
    # Create documents for Chroma
    from langchain.schema import Document
    
    # Convert all documents first
    logger.info(f"Converting {len(documents)} documents...")
    langchain_docs = [
        Document(page_content=doc["page_content"], metadata=doc["metadata"]) 
        for doc in documents
    ]
    
    # Create the Chroma database in batches to handle large datasets
    batch_size = 500
    logger.info(f"Creating Chroma database with {len(langchain_docs)} documents in batches of {batch_size}")
    
    db = None
    for i in range(0, len(langchain_docs), batch_size):
        batch = langchain_docs[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(langchain_docs) + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
        
        if db is None:
            # Create initial database with first batch
            db = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
        else:
            # Add documents to existing database
            db.add_documents(batch)
    
    # Persist the database (although Chroma should auto-persist)
    logger.info("Persisting database...")
    db.persist()
    
    logger.info(f"Chroma database created with {len(documents)} documents")
    return db

def main():
    # Configuration
    data_dir = "data/ru_rag_test_dataset-main"
    files_dir = os.path.join(data_dir, "files")
    dataset_path = os.path.join(data_dir, "ru_rag_test_dataset.pkl")
    persist_directory = "data/chroma_db"
    collection_name = "ru_rag_collection"
    chunk_size = 800
    chunk_overlap = 400
    
    # Process text files
    documents = process_text_files(files_dir, chunk_size, chunk_overlap)
    
    # Use all documents
    logger.info(f"Processing all {len(documents)} documents")
    
    # Create Chroma database
    db = create_chroma_db(documents, persist_directory, collection_name)
    
    # Load dataset for validation
    dataset = load_ru_rag_dataset(dataset_path)
    logger.info(f"Dataset loaded with {len(dataset)} entries")
    
    logger.info("Chroma database creation complete!")

if __name__ == "__main__":
    main()
