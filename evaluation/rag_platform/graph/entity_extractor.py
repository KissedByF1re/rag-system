"""Entity extraction for GraphRAG knowledge graph construction."""

import json
import re
from typing import Dict, List, Tuple, Optional

from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from rag_platform.core.base import BaseLoader


class Entity:
    """Represents an extracted entity."""
    
    def __init__(self, name: str, type: str, description: str = ""):
        self.name = name.strip()
        self.type = type.strip()
        self.description = description.strip()
    
    def __repr__(self):
        return f"Entity(name='{self.name}', type='{self.type}')"
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type, 
            "description": self.description
        }


class Relationship:
    """Represents a relationship between entities."""
    
    def __init__(self, source: str, target: str, relation: str, description: str = ""):
        self.source = source.strip()
        self.target = target.strip()
        self.relation = relation.strip()
        self.description = description.strip()
    
    def __repr__(self):
        return f"Relationship('{self.source}' --{self.relation}--> '{self.target}')"
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "description": self.description
        }


class EntityExtractor:
    """Extracts entities and relationships from documents using LLM."""
    
    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        temperature: float = 0.1,
        max_entities_per_chunk: int = 20,
        **kwargs
    ):
        """Initialize entity extractor.
        
        Args:
            model: LLM model for entity extraction
            temperature: Temperature for extraction
            max_entities_per_chunk: Maximum entities to extract per chunk
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature, **kwargs)
        self.max_entities_per_chunk = max_entities_per_chunk
        
        # Prompt template for entity extraction
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "Extract entities and relationships from this text:\\n\\n{text}")
        ])
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for entity extraction."""
        return f"""You are an expert at extracting entities and relationships from text.

Extract up to {self.max_entities_per_chunk} of the most important entities and their relationships.

For each entity, identify:
- name: The entity name (normalize to canonical form)
- type: Entity type (PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, etc.)
- description: Brief description of the entity

For relationships, identify:
- source: Source entity name
- target: Target entity name  
- relation: Relationship type (e.g., "works_for", "located_in", "part_of", "causes")
- description: Brief description of the relationship

Return ONLY a valid JSON object with this structure:
{{{{
  "entities": [
    {{{{"name": "Entity Name", "type": "TYPE", "description": "Description"}}}}
  ],
  "relationships": [
    {{{{"source": "Entity1", "target": "Entity2", "relation": "relation_type", "description": "Description"}}}}
  ]
}}}}

Focus on the most important entities and clear relationships. Normalize entity names consistently."""
    
    def extract_from_document(self, document: Document) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from a single document.
        
        Args:
            document: Document to process
            
        Returns:
            Tuple of (entities, relationships)
        """
        try:
            # Get extraction from LLM
            chain = self.extraction_prompt | self.llm
            response = chain.invoke({"text": document.page_content})
            
            # Parse JSON response
            result = json.loads(response.content)
            
            # Convert to Entity and Relationship objects
            entities = [Entity(**entity_data) for entity_data in result.get("entities", [])]
            relationships = [Relationship(**rel_data) for rel_data in result.get("relationships", [])]
            
            return entities, relationships
            
        except json.JSONDecodeError as e:
            print(f"WARNING: Failed to parse JSON from LLM response: {e}")
            return [], []
        except Exception as e:
            print(f"WARNING: Entity extraction failed: {e}")
            return [], []
    
    def extract_from_documents(self, documents: List[Document]) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from multiple documents.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Tuple of (all_entities, all_relationships)
        """
        all_entities = []
        all_relationships = []
        
        for i, doc in enumerate(documents):
            print(f"Extracting entities from document {i+1}/{len(documents)}")
            
            entities, relationships = self.extract_from_document(doc)
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        
        # Deduplicate entities by name
        unique_entities = self._deduplicate_entities(all_entities)
        
        return unique_entities, all_relationships
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities, keeping the one with the best description."""
        entity_map = {}
        
        for entity in entities:
            key = entity.name.lower()
            if key not in entity_map or len(entity.description) > len(entity_map[key].description):
                entity_map[key] = entity
        
        return list(entity_map.values())
    
    def get_entity_types(self, entities: List[Entity]) -> Dict[str, int]:
        """Get count of entities by type."""
        type_counts = {}
        for entity in entities:
            type_counts[entity.type] = type_counts.get(entity.type, 0) + 1
        return type_counts