"""Enhanced retriever with better search strategies."""

import logging
from typing import List, Dict, Optional
import numpy as np
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class EnhancedRetriever:
    """Advanced retriever with query expansion and adaptive k."""
    
    def __init__(self, 
                 persist_directory: str = "./data/chroma_db",
                 collection_name: str = "ru_rag_collection",
                 base_k: int = 10,
                 min_score_threshold: float = 0.5):
        
        self.base_k = base_k
        self.min_score_threshold = min_score_threshold
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="deepvk/USER-base",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load ChromaDB
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        
    def hybrid_search(self, query: str, k: int = None) -> List[tuple]:
        """Perform hybrid search with query expansion."""
        if k is None:
            k = self.base_k
            
        # Collect all results with scores
        all_results = {}
        
        for exp_query in [query]:
            # Vector search with scores
            vector_results = self.db.similarity_search_with_score(exp_query, k=k)
            
            for doc, score in vector_results:
                doc_id = f"{doc.metadata.get('source', '')}_{doc.metadata.get('chunk_id', '')}"
                
                if doc_id not in all_results:
                    all_results[doc_id] = {
                        'doc': doc,
                        'scores': [],
                        'queries': []
                    }
                
                all_results[doc_id]['scores'].append(score)
                all_results[doc_id]['queries'].append(exp_query)
        
        # Combine scores with boost for multiple query matches
        combined_results = []
        for doc_id, info in all_results.items():
            # Average score with boost for multiple matches
            avg_score = np.mean(info['scores'])
            match_boost = min(len(info['queries']) / len([query]), 1.0)
            combined_score = avg_score * (1 + match_boost * 0.5)
            
            combined_results.append((info['doc'], combined_score))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results[:k]
    
    def filter_by_relevance(self, documents: List[tuple], threshold: float = None) -> List[Document]:
        """Filter documents by relevance score threshold."""
        if threshold is None:
            threshold = self.min_score_threshold
            
        filtered = []
        for doc, score in documents:
            if score >= threshold:
                filtered.append(doc)
        
        return filtered
    
    def deduplicate_results(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate or near-duplicate chunks."""
        unique_docs = []
        seen_content = set()
        
        for doc in documents:
            # Create a content fingerprint
            content_fingerprint = doc.page_content[:100].strip().lower()
            
            if content_fingerprint not in seen_content:
                unique_docs.append(doc)
                seen_content.add(content_fingerprint)
        
        return unique_docs
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Main retrieval method with all enhancements."""
        logger.info(f"Retrieving for query: {query}")
        
        # 1. Hybrid search with expanded k
        scored_candidates = self.hybrid_search(query, k=self.base_k)
        
        # 2. Filter by relevance
        relevant_docs = self.filter_by_relevance(scored_candidates)
        
        # 3. Deduplicate
        unique_docs = self.deduplicate_results(relevant_docs)
        
        # 4. Return top k
        return unique_docs[:k]
    
    def retrieve_with_metadata_filter(self, query: str, metadata_filter: Dict = None, k: int = 5) -> List[Document]:
        """Retrieve with metadata filtering."""
        if metadata_filter:
            # Use Chroma's metadata filtering
            search_kwargs = {"k": self.base_k, "filter": metadata_filter}
        else:
            search_kwargs = {"k": self.base_k}
            
        results = self.db.similarity_search(query, **search_kwargs)
        
        # Deduplicate and return top k
        unique_docs = self.deduplicate_results(results)
        return unique_docs[:k]