import math
import re
import logging

import numexpr
from langchain_core.tools import BaseTool, tool

# Import the enhanced retriever
from agents.enhanced_retriever import EnhancedRetriever

logger = logging.getLogger(__name__)


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


# Format retrieved documents with relevance info
def format_contexts(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        formatted.append(f"[Источник {i}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


# Global retriever instance to avoid re-initialization
_retriever_instance = None

def load_enhanced_retriever():
    """Load the enhanced retriever with singleton pattern."""
    global _retriever_instance
    
    if _retriever_instance is None:
        logger.info("Initializing enhanced retriever...")
        _retriever_instance = EnhancedRetriever(
            persist_directory="./data/chroma_db",
            collection_name="ru_rag_collection",
            base_k=10,  # Retrieve more candidates for better filtering
            min_score_threshold=0.5
        )
        logger.info("Enhanced retriever initialized")
    
    return _retriever_instance


def database_search_func(query: str) -> str:
    """Поиск информации в базе знаний на русском языке с улучшенным ранжированием.
    
    Args:
        query: Поисковый запрос на русском языке
        
    Returns:
        Релевантная информация из базы данных
    """
    # Get the enhanced retriever
    retriever = load_enhanced_retriever()

    # Search the database for relevant documents
    documents = retriever.retrieve(query, k=5)

    # Format the documents into a string
    context_str = format_contexts(documents)
    
    # Add query understanding hint
    if not documents:
        context_str = "По данному запросу информация не найдена в базе знаний."
    elif len(documents) < 3:
        context_str += "\n\n[Примечание: Найдено мало релевантных документов. Возможно, информация по данной теме ограничена.]"

    return context_str


database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"