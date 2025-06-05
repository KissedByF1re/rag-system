"""RAGAS evaluation implementation."""

import os
from typing import Any, Dict, List

import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_platform.core.base import BaseEvaluator
from rag_platform.core.config import Settings
from rag_platform.core.registry import registry


class RAGASEvaluator(BaseEvaluator):
    """RAGAS evaluation wrapper."""
    
    def __init__(self, metrics: List[str] = None):
        if metrics is None:
            metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        
        self.metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        
        self.metrics = [self.metric_map[m] for m in metrics if m in self.metric_map]
    
    def evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate RAG results using RAGAS."""
        # Get settings for API key
        settings = Settings()
        
        # Set up OpenAI for RAGAS
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        
        # Create RAGAS-compatible LLM and embeddings
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=settings.openai_api_key
        )
        
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key
        )
        
        eval_data = []
        
        for result in results:
            contexts = []
            for doc in result.get("source_documents", []):
                contexts.append(doc.page_content)
            
            # Handle both "query" and "question" keys
            question = result.get("question", result.get("query", ""))
            
            eval_data.append({
                "question": question,  # For faithfulness and answer_relevancy
                "user_input": question,  # For context_precision and context_recall
                "answer": result["answer"],
                "contexts": contexts,  # For faithfulness and answer_relevancy
                "retrieved_contexts": contexts,  # For context_precision and context_recall
                "ground_truth": result.get("ground_truth", ""),  # For faithfulness and answer_relevancy
                "reference": result.get("ground_truth", ""),  # For context_precision and context_recall
            })
        
        dataset = Dataset.from_pandas(pd.DataFrame(eval_data))
        
        # Run evaluation with configured LLM and embeddings
        eval_results = evaluate(
            dataset, 
            metrics=self.metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False  # Don't fail completely on individual errors
        )
        
        # Extract metrics properly from RAGAS results
        metrics_dict = {}
        
        # RAGAS v0.2+ returns EvaluationResult object
        if hasattr(eval_results, 'to_pandas'):
            # Convert to pandas DataFrame to extract metrics
            df = eval_results.to_pandas()
            for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if metric in df.columns:
                    # Calculate mean of the metric column, handling NaN values
                    metric_values = df[metric].dropna()
                    if len(metric_values) > 0:
                        metrics_dict[metric] = float(metric_values.mean())
                    else:
                        metrics_dict[metric] = 0.0
                else:
                    metrics_dict[metric] = 0.0
        
        # Fallback: try to extract from _scores_dict if available
        elif hasattr(eval_results, '_scores_dict'):
            scores_dict = eval_results._scores_dict
            for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if metric in scores_dict:
                    values = scores_dict[metric]
                    if isinstance(values, list) and len(values) > 0:
                        metrics_dict[metric] = float(sum(values) / len(values))
                    elif isinstance(values, (int, float)):
                        metrics_dict[metric] = float(values)
                    else:
                        metrics_dict[metric] = 0.0
                else:
                    metrics_dict[metric] = 0.0
        
        # Fallback: direct dictionary access
        else:
            for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if metric in eval_results:
                    value = eval_results[metric]
                    if isinstance(value, (int, float)):
                        metrics_dict[metric] = float(value)
                    elif hasattr(value, 'mean'):
                        metrics_dict[metric] = float(value.mean())
                    elif isinstance(value, list) and len(value) > 0:
                        metrics_dict[metric] = float(sum(value) / len(value))
                    else:
                        metrics_dict[metric] = 0.0
                else:
                    metrics_dict[metric] = 0.0
        
        return metrics_dict


registry.register_evaluator("ragas", RAGASEvaluator)