import pandas as pd
from typing import Callable, List
from tqdm import tqdm
import torch
from dotenv import load_dotenv
import logging
import sys
from pathlib import Path
import json

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

# Load env params
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        db_path: str,
        collection_name: str,
        generate_fn: Callable[[str, List[str]], str],
        top_k: int = 3,
        model_name: str = "sentence-transformers/msmarco-bert-base-dot-v5",
    ):
        logger.info(
            f"Initializing Evaluator with db_path={db_path}, collection_name={collection_name}, top_k={top_k}"
        )
        self.client = chromadb.PersistentClient(path=db_path)

        logger.info(f"Loading collection '{collection_name}' from ChromaDB")
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=SentenceTransformerEmbeddingFunction(
                model_name=model_name,
            ),
        )
        logger.info("Collection loaded successfully")

        self.generate_fn = generate_fn
        self.top_k = top_k

    def retrieve_contexts(self, query: str) -> List[str]:
        """Retrieve top-k contexts from ChromaDB for the given query"""
        logger.debug(
            f"Querying contexts for question: '{query}' with top_k={self.top_k}"
        )
        results = self.collection.query(query_texts=[query], n_results=self.top_k)
        contexts = results["documents"][0]
        logger.debug(f"Retrieved {len(contexts)} contexts")
        return contexts

    def evaluate(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Main evaluation loop over the dataset"""
        logger.info("Starting evaluation loop")
        records = []

        for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Evaluating"):
            question = row["Вопрос"]
            reference = row["Правильный ответ"]
            logger.info(f"Evaluating sample {idx+1}/{len(dataset)}: '{question}'")

            contexts = self.retrieve_contexts(question)
            logger.debug(f"Contexts for sample {idx+1}: {contexts}")

            logger.debug("Generating answer using the generation function")
            generated = self.generate_fn(question, contexts)
            logger.info(f"Generated answer for sample {idx+1}")

            records.append(
                {
                    "question": question,
                    "reference_answer": reference,
                    "generated_answer": generated,
                    "contexts": contexts,
                }
            )

            if idx == 30:
                break

        logger.info("Evaluation completed")
        return pd.DataFrame(records)


class TransformerGenerator:
    def __init__(
        self,
        model_name: str = "ai-forever/rugpt3medium_based_on_gpt2",
        device: str = None,
    ):
        logger.info(f"Initializing TransformerGenerator with model_name={model_name}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load config to decide model type
        config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if getattr(config, "is_encoder_decoder", False):
            logger.info("Detected encoder-decoder model, using AutoModelForSeq2SeqLM")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            logger.info("Detected decoder-only model, using AutoModelForCausalLM")
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.to(self.device)
        logger.info(f"Model loaded on device {self.device}")

    def __call__(self, question: str, contexts: List[str]) -> str:
        prompt = f"Question: {question}\nContext:\n" + "\n".join(contexts) + "\nAnswer:"
        logger.debug(f"Constructed prompt: {prompt}")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"Decoded answer: {answer}")
        return answer


def main():
    logger.info("Script started")
    chromadb_path = "./data/chroma_db"
    dataset_path = "./data/ru_rag_test_dataset-main/ru_rag_test_dataset.pkl"
    model = "t-bank-ai/ruDialoGPT-small"
    results_path = "./data/results_evalution"
    json_output_path = f"{results_path}/evaluation_results.json"

    logger.info(f"Loading dataset from {dataset_path}")
    dataset = pd.read_pickle(dataset_path)

    gen = TransformerGenerator()

    evaluator = Evaluator(
        db_path=chromadb_path,
        collection_name="ru_rag_collection",
        generate_fn=gen,
        top_k=3,
        model_name=model,
    )

    results_df = evaluator.evaluate(dataset)

    Path(results_path).mkdir(parents=True, exist_ok=True)

    logger.info("Saving evaluation results to JSON")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(results_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    logger.info("Script finished successfully")


if __name__ == "__main__":
    main()
