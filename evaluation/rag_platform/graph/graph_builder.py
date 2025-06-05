"""Knowledge graph construction for GraphRAG."""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

import networkx as nx
from langchain.schema import Document

from .entity_extractor import Entity, Relationship, EntityExtractor


class KnowledgeGraph:
    """Knowledge graph built from extracted entities and relationships."""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.entities = {}  # name -> Entity
        self.communities = {}  # community_id -> community_data
        self._community_summaries = {}
    
    def add_entity(self, entity: Entity) -> None:
        """Add entity to the graph."""
        self.entities[entity.name] = entity
        self.graph.add_node(
            entity.name,
            type=entity.type,
            description=entity.description
        )
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add relationship to the graph."""
        # Ensure both entities exist
        if relationship.source not in self.graph:
            self.graph.add_node(relationship.source)
        if relationship.target not in self.graph:
            self.graph.add_node(relationship.target)
        
        # Add edge
        self.graph.add_edge(
            relationship.source,
            relationship.target,
            relation=relationship.relation,
            description=relationship.description
        )
    
    def get_neighbors(self, entity_name: str, max_depth: int = 1) -> Set[str]:
        """Get neighboring entities up to max_depth."""
        if entity_name not in self.graph:
            return set()
        
        neighbors = set()
        current_level = {entity_name}
        
        for depth in range(max_depth):
            next_level = set()
            for node in current_level:
                node_neighbors = set(self.graph.neighbors(node))
                next_level.update(node_neighbors)
                neighbors.update(node_neighbors)
            current_level = next_level
        
        return neighbors
    
    def get_subgraph(self, entities: Set[str]) -> nx.Graph:
        """Get subgraph containing specified entities."""
        return self.graph.subgraph(entities)
    
    def get_connected_components(self) -> List[Set[str]]:
        """Get connected components of the graph."""
        return [set(component) for component in nx.connected_components(self.graph)]
    
    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Get entity by name."""
        return self.entities.get(name)
    
    def save(self, file_path: str) -> None:
        """Save knowledge graph to file."""
        data = {
            "graph": nx.node_link_data(self.graph),
            "entities": {name: entity.to_dict() for name, entity in self.entities.items()},
            "communities": self.communities,
            "community_summaries": self._community_summaries
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, file_path: str) -> 'KnowledgeGraph':
        """Load knowledge graph from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        kg = cls()
        kg.graph = nx.node_link_graph(data["graph"])
        kg.entities = {name: Entity(**entity_data) for name, entity_data in data["entities"].items()}
        kg.communities = data.get("communities", {})
        kg._community_summaries = data.get("community_summaries", {})
        
        return kg


class GraphBuilder:
    """Builds knowledge graphs from documents using entity extraction."""
    
    def __init__(
        self,
        entity_extractor: Optional[EntityExtractor] = None,
        community_algorithm: str = "leiden",
        resolution: float = 1.0
    ):
        """Initialize graph builder.
        
        Args:
            entity_extractor: Entity extractor to use
            community_algorithm: Community detection algorithm
            resolution: Resolution parameter for community detection
        """
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.community_algorithm = community_algorithm
        self.resolution = resolution
    
    def build_from_documents(self, documents: List[Document]) -> KnowledgeGraph:
        """Build knowledge graph from documents.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Constructed knowledge graph
        """
        print(f"Building knowledge graph from {len(documents)} documents...")
        
        # Extract entities and relationships
        print("Extracting entities and relationships...")
        entities, relationships = self.entity_extractor.extract_from_documents(documents)
        
        print(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
        
        # Build graph
        kg = KnowledgeGraph()
        
        # Add entities
        for entity in entities:
            kg.add_entity(entity)
        
        # Add relationships
        for relationship in relationships:
            kg.add_relationship(relationship)
        
        # Detect communities
        print("Detecting communities...")
        communities = self._detect_communities(kg.graph)
        kg.communities = communities
        
        print(f"Detected {len(communities)} communities")
        
        return kg
    
    def _detect_communities(self, graph: nx.Graph) -> Dict[int, Dict]:
        """Detect communities in the graph using specified algorithm."""
        if self.community_algorithm == "leiden":
            return self._leiden_communities(graph)
        elif self.community_algorithm == "louvain":
            return self._louvain_communities(graph)
        else:
            # Simple connected components fallback
            return self._connected_component_communities(graph)
    
    def _leiden_communities(self, graph: nx.Graph) -> Dict[int, Dict]:
        """Detect communities using Leiden algorithm."""
        try:
            import community as community_louvain  # python-louvain
            
            # Convert to undirected if needed
            if graph.is_directed():
                graph = graph.to_undirected()
            
            # Use Louvain as approximation for Leiden (needs separate package for true Leiden)
            partition = community_louvain.best_partition(graph, resolution=self.resolution)
            
            # Group nodes by community
            communities = defaultdict(list)
            for node, community_id in partition.items():
                communities[community_id].append(node)
            
            # Convert to desired format
            result = {}
            for i, (community_id, nodes) in enumerate(communities.items()):
                result[i] = {
                    "id": i,
                    "nodes": nodes,
                    "size": len(nodes),
                    "algorithm": "louvain"  # Approximation
                }
            
            return result
            
        except ImportError:
            print("WARNING: python-louvain not installed, using connected components")
            return self._connected_component_communities(graph)
    
    def _louvain_communities(self, graph: nx.Graph) -> Dict[int, Dict]:
        """Detect communities using Louvain algorithm."""
        return self._leiden_communities(graph)  # Same implementation for now
    
    def _connected_component_communities(self, graph: nx.Graph) -> Dict[int, Dict]:
        """Use connected components as simple community detection."""
        components = list(nx.connected_components(graph))
        
        result = {}
        for i, component in enumerate(components):
            result[i] = {
                "id": i,
                "nodes": list(component),
                "size": len(component),
                "algorithm": "connected_components"
            }
        
        return result
    
    def update_graph(self, kg: KnowledgeGraph, new_documents: List[Document]) -> KnowledgeGraph:
        """Update existing knowledge graph with new documents.
        
        Args:
            kg: Existing knowledge graph
            new_documents: New documents to add
            
        Returns:
            Updated knowledge graph
        """
        # Extract from new documents
        entities, relationships = self.entity_extractor.extract_from_documents(new_documents)
        
        # Add to existing graph
        for entity in entities:
            kg.add_entity(entity)
        
        for relationship in relationships:
            kg.add_relationship(relationship)
        
        # Re-detect communities
        communities = self._detect_communities(kg.graph)
        kg.communities = communities
        
        return kg