"""Experiment runner for RAG pipelines."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from rag_platform.core.config import ExperimentConfig
from rag_platform.core.registry import registry


class ExperimentRunner:
    """Run RAG experiments based on configuration."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._setup_components()
    
    def _setup_components(self):
        """Initialize all components based on configuration."""
        # Initialize loader
        loader_class = registry.get_loader(self.config.data.loader)
        self.loader = loader_class()
        
        # Initialize chunker
        chunker_class = registry.get_chunker(self.config.chunking.strategy)
        self.chunker = chunker_class(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
        )
        
        # Initialize embedding
        embedding_class = registry.get_embedding(self.config.embedding.provider)
        self.embedding = embedding_class(model=self.config.embedding.model)
        
        # Initialize vector store
        vectorstore_class = registry.get_vectorstore(self.config.vectorstore.type)
        self.vectorstore = vectorstore_class(
            embedding_function=self.embedding,
            collection_name=self.config.vectorstore.collection_name,
            persist_directory=self.config.vectorstore.persist_directory,
        )
        
        # Initialize evaluator
        evaluator_name = getattr(self.config.evaluation, 'evaluator', 'ragas')
        evaluator_class = registry.get_evaluator(evaluator_name)
        self.evaluator = evaluator_class(metrics=self.config.evaluation.metrics)
    
    def run(self) -> Dict[str, Any]:
        """Run the experiment."""
        print(f"Starting experiment: {self.config.name}")
        
        # Load documents
        print("Loading documents...")
        documents = self.loader.load(self.config.data.path)
        print(f"Loaded {len(documents)} documents")
        
        # Chunk documents
        print("Chunking documents...")
        chunks = self.chunker.chunk(documents)
        print(f"Created {len(chunks)} chunks")
        
        # Add to vector store
        print("Adding to vector store...")
        self.vectorstore.add_documents(chunks)
        
        # Create retriever based on type
        retriever = self._create_retriever(chunks)
        
        # Initialize RAG chain
        chain_class = registry.get_chain(self.config.rag_chain.type)
        rag_chain = self._create_rag_chain(chain_class, retriever)
        
        # Run evaluation
        print("Running evaluation...")
        results = self._evaluate(documents, rag_chain)
        
        # Save results
        output_path = self._save_results(results)
        
        return {
            "experiment_name": self.config.name,
            "timestamp": datetime.now().isoformat(),
            "evaluation_results": results["evaluation_metrics"],
            "output_path": output_path,
        }
    
    def _create_retriever(self, chunks: List[Any]) -> Any:
        """Create retriever based on configuration."""
        retriever_type = self.config.retriever.type
        
        if retriever_type == "vector":
            return self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.retriever.k}
            )
        
        elif retriever_type == "hybrid":
            # Create hybrid retriever
            vector_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.retriever.k * 2}  # Get more for fusion
            )
            from rag_platform.retrievers.hybrid_retriever import HybridRetriever
            from rag_platform.retrievers.langchain_adapter import LangChainRetrieverAdapter
            hybrid_retriever = HybridRetriever(vector_retriever=vector_retriever)
            return LangChainRetrieverAdapter(hybrid_retriever, k=self.config.retriever.k)
        
        elif retriever_type == "reranker":
            # Create reranker retriever
            vector_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.retriever.k * 3}  # Get more for reranking
            )
            from rag_platform.retrievers.reranker_retriever import RerankerRetriever
            from rag_platform.retrievers.langchain_adapter import LangChainRetrieverAdapter
            reranker_retriever = RerankerRetriever(base_retriever=vector_retriever)
            return LangChainRetrieverAdapter(reranker_retriever, k=self.config.retriever.k)
        
        elif retriever_type == "graph":
            # Create GraphRAG retriever
            return self._create_graph_retriever(chunks)
        
        else:
            # Fallback to vector
            return self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.retriever.k}
            )
    
    def _create_graph_retriever(self, chunks: List[Any]) -> Any:
        """Create GraphRAG retriever with knowledge graph."""
        print("🔗 Building knowledge graph for GraphRAG...")
        
        # Import GraphRAG components
        from rag_platform.graph.entity_extractor import EntityExtractor
        from rag_platform.graph.graph_builder import GraphBuilder, KnowledgeGraph
        from rag_platform.graph.graph_retriever import GraphRetriever
        
        # Initialize entity extractor
        entity_extractor = EntityExtractor(
            model=self.config.llm.model,
            temperature=0.1
        )
        
        # Build knowledge graph
        graph_builder = GraphBuilder()
        knowledge_graph = KnowledgeGraph()
        
        print(f"📝 Extracting entities from {len(chunks)} chunks...")
        for i, chunk in enumerate(tqdm(chunks, desc="Extracting entities")):
            try:
                # Extract entities and relationships
                from langchain.schema import Document
                chunk_doc = Document(page_content=chunk.page_content, metadata=chunk.metadata)
                entities, relationships = entity_extractor.extract_from_document(chunk_doc)
                
                # Add to knowledge graph
                for entity in entities:
                    knowledge_graph.add_entity(entity)
                
                for relationship in relationships:
                    knowledge_graph.add_relationship(relationship)
                    
            except Exception as e:
                print(f"⚠️ Error processing chunk {i}: {e}")
                continue
        
        print(f"📊 Knowledge graph built: {len(knowledge_graph.entities)} entities, {knowledge_graph.graph.number_of_edges()} relationships")
        
        # Detect communities
        print("🏘️ Detecting communities...")
        communities = graph_builder._detect_communities(knowledge_graph.graph)
        knowledge_graph.communities = communities
        
        # Create graph retriever
        graph_config = getattr(self.config, 'graph', None)
        search_strategy = "hybrid"
        if graph_config and hasattr(graph_config, 'search_strategy'):
            search_strategy = graph_config.search_strategy
        
        return GraphRetriever(
            knowledge_graph=knowledge_graph,
            search_strategy=search_strategy,
            local_search_enabled=True,
            global_search_enabled=True,
            summarizer_model=self.config.llm.model
        )
    
    def _create_rag_chain(self, chain_class: Any, retriever: Any) -> Any:
        """Create RAG chain with proper parameters."""
        if self.config.rag_chain.type == "graph_rag":
            # GraphRAG chain needs different parameters
            return chain_class(
                retriever=retriever,
                llm_model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                prompt_template=getattr(self.config.rag_chain, 'prompt_template', 'russian_graph_qa')
            )
        else:
            # Standard chains
            return chain_class(
                retriever=retriever,
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )
    
    def _evaluate(self, documents: List[Any], rag_chain: Any) -> Dict[str, Any]:
        """Evaluate the RAG chain with complete data collection."""
        # Extract test questions from documents
        test_data = []
        for doc in documents:
            if "question" in doc.metadata and "answer" in doc.metadata:
                test_data.append({
                    "question": doc.metadata["question"],
                    "ground_truth": doc.metadata["answer"],
                    "original_context": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "row_index": doc.metadata.get("row_index", -1),
                })
        
        # Limit to sample size if specified
        if self.config.evaluation.sample_size:
            test_data = test_data[:self.config.evaluation.sample_size]
        
        print(f"Evaluating {len(test_data)} questions...")
        
        # Run RAG chain on test data with complete tracking
        complete_results = []
        for i, item in enumerate(tqdm(test_data, desc="Processing queries")):
            start_time = time.time()
            
            # Run RAG chain
            rag_result = rag_chain.run(item["question"])
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create complete result record
            complete_result = {
                # Question and answers
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "generated_answer": rag_result["answer"],
                
                # Retrieved contexts (from RAG chain)
                "retrieved_contexts": [doc.page_content for doc in rag_result.get("source_documents", [])],
                "context_sources": [doc.metadata.get("source", "unknown") for doc in rag_result.get("source_documents", [])],
                
                # Original context from dataset
                "original_context": item["original_context"],
                
                # Metadata
                "source_file": item["source"],
                "dataset_index": item["row_index"],
                "query_index": i,
                
                # Performance data
                "processing_time": processing_time,
                "num_retrieved_contexts": len(rag_result.get("source_documents", [])),
                
                # Configuration info
                "experiment_config": {
                    "llm_model": self.config.llm.model,
                    "embedding_model": self.config.embedding.model,
                    "chunk_size": self.config.chunking.chunk_size,
                    "retrieval_k": self.config.retriever.k,
                    "temperature": self.config.llm.temperature,
                }
            }
            
            complete_results.append(complete_result)
        
        # Prepare data for RAGAS evaluation
        ragas_data = []
        for result in complete_results:
            ragas_data.append({
                "question": result["question"],
                "answer": result["generated_answer"],
                "contexts": result["retrieved_contexts"],
                "ground_truth": result["ground_truth"],
            })
        
        # Evaluate with RAGAS
        print("Running RAGAS evaluation...")
        try:
            evaluation_metrics = self.evaluator.evaluate(ragas_data)
        except Exception as e:
            print(f"WARNING: RAGAS evaluation failed: {e}")
            # Continue with basic metrics
            evaluation_metrics = {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "ragas_error": str(e)
            }
        
        # Add individual RAGAS scores to complete results
        if hasattr(self.evaluator, 'evaluate_individual'):
            individual_scores = self.evaluator.evaluate_individual(ragas_data)
            for i, scores in enumerate(individual_scores):
                if i < len(complete_results):
                    complete_results[i]["ragas_scores"] = scores
        
        return {
            "evaluation_metrics": evaluation_metrics,
            "complete_results": complete_results,  # ALL results, not just first 10
            "total_samples": len(complete_results),
            "average_processing_time": sum(r["processing_time"] for r in complete_results) / len(complete_results),
            "total_processing_time": sum(r["processing_time"] for r in complete_results),
        }
    
    def _save_results(self, results: Dict[str, Any]) -> str:
        """Save experiment results."""
        # Create output directory
        output_dir = Path(self.config.output.results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.name}_{timestamp}"
        
        # Save based on format
        if self.config.output.format == "json":
            output_path = output_dir / f"{filename}.json"
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
        elif self.config.output.format == "csv":
            output_path = output_dir / f"{filename}.csv"
            df = pd.DataFrame(results["sample_results"])
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {self.config.output.format}")
        
        return str(output_path)