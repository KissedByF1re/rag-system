"""Tests for GraphRAG components."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

from rag_platform.graph.entity_extractor import Entity, Relationship, EntityExtractor
from rag_platform.graph.graph_builder import KnowledgeGraph, GraphBuilder
from rag_platform.graph.graph_retriever import GraphRetriever
from rag_platform.chains.graph_rag_chain import GraphRAGChain


class TestEntity:
    """Test Entity class."""
    
    def test_entity_creation(self):
        """Test entity creation and methods."""
        entity = Entity("Python", "PROGRAMMING_LANGUAGE", "A high-level programming language")
        
        assert entity.name == "Python"
        assert entity.type == "PROGRAMMING_LANGUAGE"
        assert entity.description == "A high-level programming language"
    
    def test_entity_to_dict(self):
        """Test entity serialization."""
        entity = Entity("Python", "PROGRAMMING_LANGUAGE", "A high-level programming language")
        data = entity.to_dict()
        
        assert data["name"] == "Python"
        assert data["type"] == "PROGRAMMING_LANGUAGE"
        assert data["description"] == "A high-level programming language"


class TestRelationship:
    """Test Relationship class."""
    
    def test_relationship_creation(self):
        """Test relationship creation."""
        rel = Relationship("Python", "Machine Learning", "used_for", "Python is used for ML")
        
        assert rel.source == "Python"
        assert rel.target == "Machine Learning"
        assert rel.relation == "used_for"
        assert rel.description == "Python is used for ML"
    
    def test_relationship_to_dict(self):
        """Test relationship serialization."""
        rel = Relationship("Python", "Machine Learning", "used_for", "Python is used for ML")
        data = rel.to_dict()
        
        assert data["source"] == "Python"
        assert data["target"] == "Machine Learning"
        assert data["relation"] == "used_for"
        assert data["description"] == "Python is used for ML"


class TestEntityExtractor:
    """Test EntityExtractor class."""
    
    @patch('rag_platform.graph.entity_extractor.ChatOpenAI')
    def test_entity_extractor_init(self, mock_chat_openai):
        """Test entity extractor initialization."""
        extractor = EntityExtractor(model="gpt-4", temperature=0.1, max_entities_per_chunk=15)
        
        mock_chat_openai.assert_called_once()
        assert extractor.max_entities_per_chunk == 15
    
    @patch('rag_platform.graph.entity_extractor.ChatOpenAI')
    def test_extract_from_document_success(self, mock_chat_openai):
        """Test successful entity extraction."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """{
            "entities": [
                {"name": "Python", "type": "LANGUAGE", "description": "Programming language"},
                {"name": "Machine Learning", "type": "FIELD", "description": "AI subfield"}
            ],
            "relationships": [
                {"source": "Python", "target": "Machine Learning", "relation": "used_for", "description": "Python is used for ML"}
            ]
        }"""
        
        mock_llm = Mock()
        mock_llm.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        extractor = EntityExtractor()
        doc = Document(page_content="Python is used for machine learning applications.")
        
        # Mock the chain
        with patch.object(extractor, 'extraction_prompt') as mock_prompt:
            mock_chain = Mock()
            mock_chain.invoke.return_value = mock_response
            mock_prompt.__or__.return_value = mock_chain
            
            entities, relationships = extractor.extract_from_document(doc)
        
        assert len(entities) == 2
        assert len(relationships) == 1
        assert entities[0].name == "Python"
        assert relationships[0].source == "Python"
    
    @patch('rag_platform.graph.entity_extractor.ChatOpenAI')
    def test_extract_from_document_json_error(self, mock_chat_openai):
        """Test handling of JSON parse errors."""
        mock_response = Mock()
        mock_response.content = "Invalid JSON"
        
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        extractor = EntityExtractor()
        doc = Document(page_content="Test content")
        
        with patch.object(extractor, 'extraction_prompt') as mock_prompt:
            mock_chain = Mock()
            mock_chain.invoke.return_value = mock_response
            mock_prompt.__or__.return_value = mock_chain
            
            entities, relationships = extractor.extract_from_document(doc)
        
        # Should return empty lists on error
        assert entities == []
        assert relationships == []
    
    @patch('rag_platform.graph.entity_extractor.ChatOpenAI')
    def test_deduplicate_entities(self, mock_chat_openai):
        """Test entity deduplication."""
        mock_chat_openai.return_value = Mock()
        extractor = EntityExtractor()
        entities = [
            Entity("Python", "LANGUAGE", "Short description"),
            Entity("python", "LANGUAGE", "Much longer and better description"),
            Entity("Java", "LANGUAGE", "Another language")
        ]
        
        unique_entities = extractor._deduplicate_entities(entities)
        
        assert len(unique_entities) == 2
        # Should keep the one with better description
        python_entity = next(e for e in unique_entities if e.name.lower() == "python")
        assert python_entity.description == "Much longer and better description"


class TestKnowledgeGraph:
    """Test KnowledgeGraph class."""
    
    def test_knowledge_graph_creation(self):
        """Test knowledge graph creation."""
        kg = KnowledgeGraph()
        
        assert len(kg.entities) == 0
        assert len(kg.communities) == 0
        assert kg.graph.number_of_nodes() == 0
    
    def test_add_entity(self):
        """Test adding entities to graph."""
        kg = KnowledgeGraph()
        entity = Entity("Python", "LANGUAGE", "Programming language")
        
        kg.add_entity(entity)
        
        assert "Python" in kg.entities
        assert kg.graph.has_node("Python")
        assert kg.graph.nodes["Python"]["type"] == "LANGUAGE"
    
    def test_add_relationship(self):
        """Test adding relationships to graph."""
        kg = KnowledgeGraph()
        rel = Relationship("Python", "Machine Learning", "used_for", "Python is used for ML")
        
        kg.add_relationship(rel)
        
        assert kg.graph.has_node("Python")
        assert kg.graph.has_node("Machine Learning")
        assert kg.graph.has_edge("Python", "Machine Learning")
        assert kg.graph.edges["Python", "Machine Learning"]["relation"] == "used_for"
    
    def test_get_neighbors(self):
        """Test getting neighboring entities."""
        kg = KnowledgeGraph()
        
        # Add entities and relationships
        kg.add_relationship(Relationship("A", "B", "connects", ""))
        kg.add_relationship(Relationship("B", "C", "connects", ""))
        kg.add_relationship(Relationship("C", "D", "connects", ""))
        
        # Test depth 1
        neighbors = kg.get_neighbors("B", max_depth=1)
        assert "A" in neighbors
        assert "C" in neighbors
        assert "D" not in neighbors  # Depth 2
        
        # Test depth 2
        neighbors = kg.get_neighbors("B", max_depth=2)
        assert "A" in neighbors
        assert "C" in neighbors
        assert "D" in neighbors
    
    def test_get_entity_by_name(self):
        """Test retrieving entity by name."""
        kg = KnowledgeGraph()
        entity = Entity("Python", "LANGUAGE", "Programming language")
        kg.add_entity(entity)
        
        retrieved = kg.get_entity_by_name("Python")
        assert retrieved == entity
        
        missing = kg.get_entity_by_name("NonExistent")
        assert missing is None


class TestGraphBuilder:
    """Test GraphBuilder class."""
    
    def test_graph_builder_init(self):
        """Test graph builder initialization."""
        mock_extractor = Mock()
        builder = GraphBuilder(entity_extractor=mock_extractor, community_algorithm="leiden")
        
        assert builder.entity_extractor == mock_extractor
        assert builder.community_algorithm == "leiden"
    
    @patch('rag_platform.graph.graph_builder.EntityExtractor')
    def test_build_from_documents(self, mock_extractor_class):
        """Test building graph from documents."""
        # Mock entity extractor
        mock_extractor = Mock()
        mock_extractor.extract_from_documents.return_value = (
            [Entity("Python", "LANGUAGE", "Programming language")],
            [Relationship("Python", "AI", "used_for", "")]
        )
        mock_extractor_class.return_value = mock_extractor
        
        builder = GraphBuilder()
        documents = [Document(page_content="Python is used for AI")]
        
        # Mock community detection
        with patch.object(builder, '_detect_communities') as mock_detect:
            mock_detect.return_value = {0: {"id": 0, "nodes": ["Python", "AI"], "size": 2}}
            
            kg = builder.build_from_documents(documents)
        
        assert len(kg.entities) == 1
        assert kg.graph.has_edge("Python", "AI")
        assert len(kg.communities) == 1
    
    @patch('rag_platform.graph.graph_builder.EntityExtractor')
    def test_connected_component_communities(self, mock_extractor_class):
        """Test connected component community detection."""
        mock_extractor_class.return_value = Mock()
        builder = GraphBuilder()
        
        # Create a simple graph
        import networkx as nx
        graph = nx.Graph()
        graph.add_edges_from([("A", "B"), ("C", "D")])  # Two components
        
        communities = builder._connected_component_communities(graph)
        
        assert len(communities) == 2
        assert communities[0]["size"] == 2
        assert communities[1]["size"] == 2


class TestGraphRetriever:
    """Test GraphRetriever class."""
    
    @patch('rag_platform.graph.graph_retriever.ChatOpenAI')
    def test_graph_retriever_init(self, mock_chat_openai):
        """Test graph retriever initialization."""
        mock_chat_openai.return_value = Mock()
        mock_kg = Mock()
        retriever = GraphRetriever(
            knowledge_graph=mock_kg,
            search_strategy="hybrid",
            max_tokens=4000
        )
        
        assert retriever.knowledge_graph == mock_kg
        assert retriever.search_strategy == "hybrid"
        assert retriever.max_tokens == 4000
    
    @patch('rag_platform.graph.graph_retriever.ChatOpenAI')
    def test_extract_entities_from_query(self, mock_chat_openai):
        """Test extracting entities from query."""
        mock_kg = Mock()
        mock_kg.entities = {"Python": Mock(), "Machine Learning": Mock(), "Java": Mock()}
        
        retriever = GraphRetriever(knowledge_graph=mock_kg)
        
        # Test entity extraction
        entities = retriever._extract_entities_from_query("How is Python used in Machine Learning?")
        
        assert "Python" in entities
        assert "Machine Learning" in entities
        assert "Java" not in entities
    
    @patch('rag_platform.graph.graph_retriever.ChatOpenAI')
    def test_find_relevant_communities(self, mock_chat_openai):
        """Test finding relevant communities."""
        mock_kg = Mock()
        mock_kg.communities = {
            0: {"nodes": ["Python", "AI", "ML"]},
            1: {"nodes": ["Java", "Enterprise"]},
            2: {"nodes": ["Python", "Web", "Django"]}
        }
        
        retriever = GraphRetriever(knowledge_graph=mock_kg)
        
        communities = retriever._find_relevant_communities(["Python"])
        
        # Should return communities containing Python, sorted by relevance
        assert 0 in communities  # Contains Python
        assert 2 in communities  # Contains Python
        assert 1 not in communities or communities.index(1) > communities.index(0)


class TestGraphRAGChain:
    """Test GraphRAGChain class."""
    
    @patch('rag_platform.chains.graph_rag_chain.ChatOpenAI')
    def test_graph_rag_chain_init(self, mock_chat_openai):
        """Test GraphRAG chain initialization."""
        mock_retriever = Mock()
        chain = GraphRAGChain(
            retriever=mock_retriever,
            llm_model="gpt-4",
            temperature=0.1
        )
        
        assert chain.retriever == mock_retriever
        mock_chat_openai.assert_called_once()
    
    @patch('rag_platform.chains.graph_rag_chain.ChatOpenAI')
    def test_run_with_context(self, mock_chat_openai):
        """Test running chain with provided context."""
        mock_retriever = Mock()
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "This is a test answer."
        
        mock_chat_openai.return_value = mock_llm
        
        chain = GraphRAGChain(retriever=mock_retriever)
        
        # Mock the prompt chain
        with patch.object(chain, 'prompt') as mock_prompt:
            mock_chain_result = Mock()
            mock_chain_result.invoke.return_value = mock_response
            mock_prompt.__or__.return_value = mock_chain_result
            
            context = [Document(page_content="Test context", metadata={"source": "test"})]
            result = chain.run("Test query", context=context)
        
        assert result["answer"] == "This is a test answer."
        assert result["num_context_docs"] == 1
        assert result["retrieval_method"] == "graph"
    
    @patch('rag_platform.chains.graph_rag_chain.ChatOpenAI')
    def test_format_context(self, mock_chat_openai):
        """Test context formatting."""
        mock_retriever = Mock()
        chain = GraphRAGChain(retriever=mock_retriever)
        
        documents = [
            Document(
                page_content="Local info",
                metadata={"search_type": "local", "community_id": 1}
            ),
            Document(
                page_content="Global info",
                metadata={"search_type": "global"}
            )
        ]
        
        formatted = chain._format_context(documents)
        
        assert "[Локальная информация - Сообщество 1]" in formatted
        assert "[Глобальная информация]" in formatted
        assert "Local info" in formatted
        assert "Global info" in formatted