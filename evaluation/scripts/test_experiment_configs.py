#!/usr/bin/env python3
"""Test experiment configurations without running full experiments."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import rag_platform
from rag_platform.core.config import ExperimentConfig


def test_config(config_path: str) -> bool:
    """Test if a configuration loads correctly."""
    try:
        config = ExperimentConfig.from_yaml(config_path)
        print(f"✅ {config_path}")
        print(f"   Name: {config.name}")
        print(f"   LLM: {config.llm.model}")
        print(f"   Embeddings: {config.embedding.model}")
        print(f"   Chunk size: {config.chunking.chunk_size}")
        print(f"   Sample size: {config.evaluation.sample_size}")
        return True
    except Exception as e:
        print(f"❌ {config_path}: {e}")
        return False


def main():
    """Test all experiment configurations."""
    print("🧪 Testing Experiment Configurations\\n")
    
    configs = [
        "configs/experiments/baseline_full_wikipedia_ragas.yaml",
        "configs/experiments/enhanced_rag.yaml",
        "configs/experiments/advanced_rag.yaml"
    ]
    
    results = []
    for config_path in configs:
        if Path(config_path).exists():
            result = test_config(config_path)
            results.append(result)
        else:
            print(f"❌ {config_path}: File not found")
            results.append(False)
        print()
    
    print(f"📊 Results: {sum(results)}/{len(results)} configurations valid")
    
    if all(results):
        print("🎉 All configurations are ready for experiments!")
        print("💡 Run experiments with: python scripts/run_experiments.py")
    else:
        print("⚠️  Fix configuration errors before running experiments.")


if __name__ == "__main__":
    main()