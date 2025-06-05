"""GraphRAG chain implementation."""

from typing import Any, Dict, List, Optional

from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from rag_platform.core.base import BaseRAGChain
from rag_platform.core.registry import registry
from rag_platform.graph.graph_retriever import GraphRetriever


class GraphRAGChain(BaseRAGChain):
    """RAG chain that uses knowledge graph for enhanced retrieval and reasoning."""
    
    def __init__(
        self,
        retriever: GraphRetriever,
        llm_model: str = "gpt-4.1-nano",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        prompt_template: str = "graph_qa",
        **kwargs
    ):
        """Initialize GraphRAG chain.
        
        Args:
            retriever: Graph-based retriever
            llm_model: LLM model for generation
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
            prompt_template: Prompt template name
        """
        super().__init__(**kwargs)
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.prompt_template_name = prompt_template
        self.prompt = self._get_prompt_template(prompt_template)
    
    def _get_prompt_template(self, template_name: str) -> ChatPromptTemplate:
        """Get prompt template for GraphRAG."""
        if template_name == "graph_qa":
            return ChatPromptTemplate.from_messages([
                ("system", self._get_graph_qa_system_prompt()),
                ("human", "Question: {question}\\n\\nGraph Context:\\n{context}\\n\\nAnswer:")
            ])
        elif template_name == "russian_graph_qa":
            return ChatPromptTemplate.from_messages([
                ("system", self._get_russian_graph_qa_system_prompt()),
                ("human", "Вопрос: {question}\\n\\nКонтекст из графа знаний:\\n{context}\\n\\nОтвет:")
            ])
        else:
            # Default graph QA prompt
            return self._get_prompt_template("graph_qa")
    
    def _get_graph_qa_system_prompt(self) -> str:
        """Get system prompt for graph-based QA."""
        return """You are an expert assistant that answers questions using information from a knowledge graph.

The context provided comes from a knowledge graph that contains:
- Entities (people, places, concepts, events) with descriptions
- Relationships between entities 
- Community-based information clustering
- Both local (specific) and global (overview) information

When answering:
1. Use the graph context to provide comprehensive, well-structured answers
2. Leverage entity relationships to explain connections and implications
3. Reference specific entities and relationships when relevant
4. If the context contains both local and global information, synthesize them appropriately
5. Be precise and factual, citing the graph-based evidence
6. If information is insufficient, state what is missing rather than guessing

Focus on providing answers that demonstrate understanding of the interconnected nature of the information in the knowledge graph."""
    
    def _get_russian_graph_qa_system_prompt(self) -> str:
        """Get Russian system prompt for graph-based QA.""" 
        return """Вы эксперт-ассистент, который отвечает на вопросы, используя информацию из графа знаний.

Предоставленный контекст поступает из графа знаний, который содержит:
- Сущности (люди, места, концепции, события) с описаниями
- Связи между сущностями
- Информацию, сгруппированную по сообществам
- Как локальную (конкретную), так и глобальную (обзорную) информацию

При ответе:
1. Используйте контекст графа для предоставления исчерпывающих, хорошо структурированных ответов
2. Используйте связи между сущностями для объяснения соединений и следствий
3. Ссылайтесь на конкретные сущности и связи, когда это уместно
4. Если контекст содержит как локальную, так и глобальную информацию, синтезируйте их соответственно
5. Будьте точными и фактичными, ссылаясь на доказательства из графа
6. Если информации недостаточно, укажите что отсутствует, а не угадывайте

Сосредоточьтесь на предоставлении ответов, которые демонстрируют понимание взаимосвязанной природы информации в графе знаний."""
    
    def run(self, query: str, context: Optional[List[Document]] = None) -> Dict[str, Any]:
        """Run the GraphRAG chain.
        
        Args:
            query: User question
            context: Optional pre-retrieved context (if None, will retrieve using graph)
            
        Returns:
            Dict with answer and metadata
        """
        # Retrieve context using graph if not provided
        if context is None:
            context = self.retriever.retrieve(query, k=5)
        
        # Format context for prompt
        context_text = self._format_context(context)
        
        # Generate answer using LLM
        chain = self.prompt | self.llm
        response = chain.invoke({
            "question": query,
            "context": context_text
        })
        
        # Extract sources and metadata
        sources = self._extract_sources(context)
        graph_metadata = self._extract_graph_metadata(context)
        
        return {
            "answer": response.content,
            "sources": sources,
            "source_documents": context,  # Standard key expected by experiment runner
            "context_docs": context,     # Keep for backward compatibility
            "graph_metadata": graph_metadata,
            "retrieval_method": "graph",
            "num_context_docs": len(context)
        }
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string."""
        if not documents:
            return "Контекст не найден."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            # Get metadata for formatting
            search_type = doc.metadata.get("search_type", "unknown")
            community_id = doc.metadata.get("community_id")
            
            # Format header based on search type
            if search_type == "local" and community_id is not None:
                header = f"[Локальная информация - Сообщество {community_id}]"
            elif search_type == "global":
                header = f"[Глобальная информация]"
            else:
                header = f"[Документ {i}]"
            
            context_parts.append(f"{header}\\n{doc.page_content}")
        
        return "\\n\\n".join(context_parts)
    
    def _extract_sources(self, documents: List[Document]) -> List[str]:
        """Extract source information from documents."""
        sources = []
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            search_type = doc.metadata.get("search_type", "")
            if search_type:
                source = f"{source} ({search_type})"
            sources.append(source)
        return sources
    
    def _extract_graph_metadata(self, documents: List[Document]) -> Dict[str, Any]:
        """Extract graph-specific metadata."""
        search_types = [doc.metadata.get("search_type") for doc in documents]
        community_ids = [doc.metadata.get("community_id") for doc in documents if doc.metadata.get("community_id") is not None]
        
        return {
            "search_types": search_types,
            "communities_used": community_ids,
            "total_communities": len(set(community_ids)),
            "uses_local_search": "local" in search_types,
            "uses_global_search": "global" in search_types
        }
    
    def get_retriever(self) -> GraphRetriever:
        """Get the graph retriever."""
        return self.retriever
    
    def get_knowledge_graph(self):
        """Get the underlying knowledge graph."""
        return self.retriever.get_knowledge_graph()


# Register the chain
registry.register_chain("graph_rag", GraphRAGChain)