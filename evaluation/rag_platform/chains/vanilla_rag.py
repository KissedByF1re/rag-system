"""Vanilla RAG chain implementation."""

from typing import Any, Dict, List, Optional

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from rag_platform.core.base import BaseRAGChain
from rag_platform.core.config import Settings
from rag_platform.core.registry import registry
from rag_platform.chains.prompt_templates import RUSSIAN_QA_TEMPLATES


class VanillaRAG(BaseRAGChain):
    """Basic retrieve-and-generate RAG chain."""
    
    def __init__(
        self,
        retriever,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: int = 500,
        prompt_template: Optional[str] = None,
    ):
        settings = Settings()
        
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=settings.openai_api_key,
        )
        
        if prompt_template is None:
            prompt_template = RUSSIAN_QA_TEMPLATES["default"]
        elif prompt_template in RUSSIAN_QA_TEMPLATES:
            prompt_template = RUSSIAN_QA_TEMPLATES[prompt_template]
        
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )
    
    def run(self, query: str, context: Optional[List[Document]] = None) -> Dict[str, Any]:
        """Run the RAG chain."""
        result = self.qa_chain({"query": query})
        
        return {
            "query": query,
            "answer": result["result"],
            "source_documents": result.get("source_documents", []),
        }


registry.register_chain("vanilla", VanillaRAG)