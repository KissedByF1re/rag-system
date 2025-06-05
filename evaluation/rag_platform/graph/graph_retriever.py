"""Graph-based retrieval for GraphRAG."""

from typing import List, Set, Dict, Optional, Tuple
import math
from collections import defaultdict

from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from rag_platform.retrievers.base_retriever import BaseRetriever
from rag_platform.core.registry import registry
from .graph_builder import KnowledgeGraph


class GraphRetriever(BaseRetriever):
    """Graph-based retriever using knowledge graph for enhanced retrieval."""
    
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        search_strategy: str = "hybrid",
        local_search_enabled: bool = True,
        global_search_enabled: bool = True,
        community_level: int = 2,
        max_tokens: int = 8000,
        summarizer_model: str = "gpt-4.1-nano",
        **kwargs
    ):
        """Initialize graph retriever.
        
        Args:
            knowledge_graph: Knowledge graph for retrieval
            search_strategy: Search strategy ("local", "global", "hybrid")
            local_search_enabled: Enable local search
            global_search_enabled: Enable global search  
            community_level: Community level for local search
            max_tokens: Maximum tokens for context
            summarizer_model: Model for summarization
        """
        super().__init__(**kwargs)
        self.knowledge_graph = knowledge_graph
        self.search_strategy = search_strategy
        self.local_search_enabled = local_search_enabled
        self.global_search_enabled = global_search_enabled
        self.community_level = community_level
        self.max_tokens = max_tokens
        
        # LLM for summarization
        self.summarizer = ChatOpenAI(model=summarizer_model, temperature=0.1)
        
        # Prompts
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the following information concisely, focusing on key facts and relationships relevant to the query."),
            ("human", "Query: {query}\\n\\nInformation:\\n{information}")
        ])
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve documents using graph-based search.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of relevant documents with graph context
        """
        results = []
        
        if self.search_strategy in ["local", "hybrid"] and self.local_search_enabled:
            local_results = self._local_search(query, k)
            results.extend(local_results)
        
        if self.search_strategy in ["global", "hybrid"] and self.global_search_enabled:
            global_results = self._global_search(query, k)
            results.extend(global_results)
        
        # Deduplicate and limit results
        unique_results = self._deduplicate_results(results)
        
        # Fallback: if no results, create a basic global summary
        if not unique_results and len(self.knowledge_graph.entities) > 0:
            fallback_context = self._create_fallback_context(query)
            if fallback_context:
                fallback_doc = Document(
                    page_content=fallback_context,
                    metadata={
                        "source": "graph_fallback",
                        "search_type": "fallback", 
                        "query": query
                    }
                )
                unique_results.append(fallback_doc)
        
        return unique_results[:k]
    
    def _local_search(self, query: str, k: int) -> List[Document]:
        """Perform local search within relevant communities.
        
        Local search finds entities related to the query and retrieves
        information from their local community neighborhoods.
        """
        # Find entities mentioned in query
        query_entities = self._extract_entities_from_query(query)
        
        if not query_entities:
            return []
        
        # Get relevant communities
        relevant_communities = self._find_relevant_communities(query_entities)
        
        # Generate local context for each community
        local_contexts = []
        for community_id in relevant_communities[:self.community_level]:
            context = self._generate_community_context(community_id, query)
            if context:
                local_contexts.append(context)
        
        # Convert to documents
        documents = []
        for i, context in enumerate(local_contexts[:k]):
            doc = Document(
                page_content=context,
                metadata={
                    "source": "graph_local_search",
                    "search_type": "local",
                    "community_id": relevant_communities[i] if i < len(relevant_communities) else None,
                    "query": query
                }
            )
            documents.append(doc)
        
        return documents
    
    def _global_search(self, query: str, k: int) -> List[Document]:
        """Perform global search across the entire knowledge graph.
        
        Global search provides high-level summaries and overviews
        relevant to the query from the entire knowledge base.
        """
        # Get global summary of relevant information
        global_context = self._generate_global_context(query)
        
        if not global_context:
            return []
        
        # Create document with global context
        doc = Document(
            page_content=global_context,
            metadata={
                "source": "graph_global_search", 
                "search_type": "global",
                "query": query
            }
        )
        
        return [doc]
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract entities mentioned in the query."""
        # Enhanced entity matching with fuzzy matching and keywords
        query_lower = query.lower()
        found_entities = []
        
        # Direct entity name matching
        for entity_name in self.knowledge_graph.entities.keys():
            if entity_name.lower() in query_lower:
                found_entities.append(entity_name)
        
        # If no direct matches, try partial matching and keywords
        if not found_entities:
            # Try partial matching (entity words in query)
            for entity_name in self.knowledge_graph.entities.keys():
                entity_words = entity_name.lower().split()
                if any(word in query_lower for word in entity_words if len(word) > 3):
                    found_entities.append(entity_name)
                    
        # If still no matches, try matching entity descriptions
        if not found_entities:
            for entity_name, entity in self.knowledge_graph.entities.items():
                if hasattr(entity, 'description') and entity.description:
                    desc_words = entity.description.lower().split()
                    if any(word in query_lower for word in desc_words if len(word) > 4):
                        found_entities.append(entity_name)
                        if len(found_entities) >= 5:  # Limit to prevent too many matches
                            break
        
        return found_entities[:10]  # Limit to top 10 entities
    
    def _find_relevant_communities(self, query_entities: List[str]) -> List[int]:
        """Find communities that contain the query entities."""
        community_scores = defaultdict(int)
        
        for entity in query_entities:
            for community_id, community_data in self.knowledge_graph.communities.items():
                if entity in community_data["nodes"]:
                    community_scores[community_id] += 1
        
        # Sort by relevance score
        sorted_communities = sorted(
            community_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [community_id for community_id, _ in sorted_communities]
    
    def _generate_community_context(self, community_id: int, query: str) -> str:
        """Generate context for a specific community."""
        community_data = self.knowledge_graph.communities.get(community_id)
        if not community_data:
            return ""
        
        # Get community nodes and their information
        nodes = community_data["nodes"]
        
        # Build context from entities and relationships
        context_parts = []
        
        # Add entity information
        for node in nodes[:10]:  # Limit to avoid too much context
            entity = self.knowledge_graph.get_entity_by_name(node)
            if entity and entity.description:
                context_parts.append(f"{entity.name} ({entity.type}): {entity.description}")
        
        # Add relationship information
        subgraph = self.knowledge_graph.get_subgraph(set(nodes))
        for edge in list(subgraph.edges(data=True))[:20]:  # Limit relationships
            source, target, attrs = edge
            relation = attrs.get("relation", "related_to")
            desc = attrs.get("description", "")
            rel_info = f"{source} --{relation}--> {target}"
            if desc:
                rel_info += f" ({desc})"
            context_parts.append(rel_info)
        
        if not context_parts:
            return ""
        
        # Combine and summarize if too long
        full_context = "\\n".join(context_parts)
        
        # Estimate tokens (rough approximation)
        estimated_tokens = len(full_context.split()) * 1.3
        
        if estimated_tokens > self.max_tokens / 2:
            # Summarize the context
            chain = self.summary_prompt | self.summarizer
            summary_response = chain.invoke({
                "query": query,
                "information": full_context
            })
            return summary_response.content
        
        return full_context
    
    def _generate_global_context(self, query: str) -> str:
        """Generate global context across the entire knowledge graph."""
        try:
            # Get overview of the knowledge graph
            total_entities = len(self.knowledge_graph.entities)
            total_communities = len(self.knowledge_graph.communities)
            
            # If no entities, return empty
            if total_entities == 0:
                return ""
            
            # Get entity type distribution
            entity_types = defaultdict(int)
            for entity in self.knowledge_graph.entities.values():
                entity_type = getattr(entity, 'type', 'UNKNOWN')
                entity_types[entity_type] += 1
            
            # Build global summary
            global_info = [
                f"Knowledge base contains {total_entities} entities across {total_communities} communities.",
                f"Entity types: {dict(entity_types)}"
            ]
            
            # Add community summaries if available
            for community_id, community_data in list(self.knowledge_graph.communities.items())[:5]:
                size = community_data.get("size", len(community_data.get("nodes", [])))
                sample_nodes = community_data.get("nodes", [])[:3]
                if sample_nodes:
                    global_info.append(f"Community {community_id}: {size} entities including {', '.join(sample_nodes)}")
            
            full_context = "\\n".join(global_info)
            
            # Don't use LLM summarization for now - just return the context directly
            # This avoids potential API issues and makes debugging easier
            return full_context
            
        except Exception as e:
            print(f"WARNING: Error generating global context: {e}")
            return ""
    
    def _deduplicate_results(self, results: List[Document]) -> List[Document]:
        """Remove duplicate results based on content."""
        seen_content = set()
        unique_results = []
        
        for doc in results:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(doc)
        
        return unique_results
    
    def _create_fallback_context(self, query: str) -> str:
        """Create fallback context when no specific entities/communities are found."""
        # Get some relevant entities by type
        relevant_info = []
        
        # Add general statistics
        total_entities = len(self.knowledge_graph.entities)
        total_communities = len(self.knowledge_graph.communities)
        relevant_info.append(f"База знаний содержит {total_entities} сущностей в {total_communities} сообществах.")
        
        # Add a few sample entities of different types
        entity_samples = {}
        for entity_name, entity in list(self.knowledge_graph.entities.items())[:20]:
            entity_type = getattr(entity, 'type', 'UNKNOWN')
            if entity_type not in entity_samples:
                entity_samples[entity_type] = []
            if len(entity_samples[entity_type]) < 3:
                description = getattr(entity, 'description', '')
                if description:
                    entity_samples[entity_type].append(f"{entity_name}: {description[:100]}...")
                else:
                    entity_samples[entity_type].append(entity_name)
        
        # Add entity type information
        for entity_type, samples in entity_samples.items():
            relevant_info.append(f"\\n{entity_type}: {', '.join(samples)}")
        
        return "\\n".join(relevant_info) if relevant_info else "Информация из графа знаний недоступна."
    
    def get_knowledge_graph(self) -> KnowledgeGraph:
        """Get the underlying knowledge graph."""
        return self.knowledge_graph


# Register the retriever
registry.register_retriever("graph", GraphRetriever)