#!/usr/bin/env python3
"""End-to-end test without external dependencies."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import main package to register all components
import rag_platform

from rag_platform.core.config import ExperimentConfig
from rag_platform.loaders import PickleLoader
from rag_platform.chunkers import RecursiveChunker
from rag_platform.vectorstores import ChromaStore


def test_data_pipeline():
    """Test the data processing pipeline."""
    print("=== Testing Data Pipeline ===")
    
    # Check if dataset exists
    dataset_path = "data/ru_rag_test_dataset.pkl"
    if not Path(dataset_path).exists():
        print("⚠️  Dataset not found. Please run: python scripts/download_dataset.py")
        return False
    
    # Test 1: Load documents
    print("1. Loading documents...")
    loader = PickleLoader()
    docs = loader.load(dataset_path)
    print(f"   ✅ Loaded {len(docs)} documents")
    
    if not docs:
        print("   ❌ No documents loaded")
        return False
    
    # Verify document structure
    sample_doc = docs[0]
    print(f"   - Sample question: {sample_doc.metadata.get('question', 'N/A')[:50]}...")
    print(f"   - Sample answer: {sample_doc.metadata.get('answer', 'N/A')[:50]}...")
    print(f"   - Content length: {len(sample_doc.page_content)} chars")
    
    # Test 2: Chunk documents
    print("\\n2. Chunking documents...")
    chunker = RecursiveChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk(docs[:10])  # Test with first 10 docs
    print(f"   ✅ Created {len(chunks)} chunks from 10 documents")
    
    if chunks:
        print(f"   - Sample chunk length: {len(chunks[0].page_content)} chars")
        print(f"   - Chunk preview: {chunks[0].page_content[:100]}...")
    
    # Test 3: Configuration loading
    print("\\n3. Testing configuration...")
    try:
        config = ExperimentConfig.from_yaml("configs/experiments/baseline_full_wikipedia_ragas.yaml")
        print(f"   ✅ Loaded config: {config.name}")
        print(f"   - Data path: {config.data.path}")
        print(f"   - Chunk size: {config.chunking.chunk_size}")
        print(f"   - LLM model: {config.llm.model}")
    except Exception as e:
        print(f"   ❌ Config loading failed: {e}")
        return False
    
    print("\\n✅ Data pipeline test completed successfully!")
    return True


def test_vector_store():
    """Test vector store initialization (without embeddings)."""
    print("\\n=== Testing Vector Store ===")
    
    try:
        # Create a mock embedding function for testing
        class MockEmbedding:
            def embed_documents(self, texts):
                # Return dummy embeddings
                return [[0.1] * 100 for _ in texts]
            
            def embed_query(self, text):
                return [0.1] * 100
        
        mock_embedding = MockEmbedding()
        
        # Test ChromaDB initialization
        store = ChromaStore(
            embedding_function=mock_embedding,
            collection_name="test_collection",
            persist_directory="./test_chroma_db"
        )
        print("   ✅ ChromaStore initialized successfully")
        
        # Cleanup test directory
        import shutil
        if Path("./test_chroma_db").exists():
            shutil.rmtree("./test_chroma_db")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Vector store test failed: {e}")
        return False


def test_cli_commands():
    """Test CLI commands."""
    print("\\n=== Testing CLI Commands ===")
    
    from rag_platform.cli import cli
    from click.testing import CliRunner
    
    runner = CliRunner()
    
    # Test help command
    result = runner.invoke(cli, ["--help"])
    if result.exit_code == 0:
        print("   ✅ CLI help command works")
    else:
        print(f"   ❌ CLI help failed: {result.output}")
        return False
    
    # Test list-components
    result = runner.invoke(cli, ["list-components"])
    if result.exit_code == 0 and "pickle_loader" in result.output:
        print("   ✅ CLI list-components works")
    else:
        print(f"   ❌ CLI list-components failed")
        return False
    
    # Test check-env
    result = runner.invoke(cli, ["check-env"])
    if result.exit_code == 0:
        print("   ✅ CLI check-env works")
    else:
        print(f"   ❌ CLI check-env failed")
        return False
    
    return True


def main():
    """Run all end-to-end tests."""
    print("🚀 Starting End-to-End Tests\\n")
    
    tests = [
        test_data_pipeline,
        test_vector_store,
        test_cli_commands,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\\n" + "="*60)
    if all(results):
        print("✅ ALL END-TO-END TESTS PASSED!")
        print("The RAG platform is ready for use.")
    else:
        print("❌ Some tests failed. Check the output above.")
        failed_count = len([r for r in results if not r])
        print(f"Failed: {failed_count}/{len(results)} tests")
        
    print("="*60)
    
    if all(results):
        print("\\n🎉 Next steps:")
        print("1. Set your OpenAI API key in .env file")
        print("2. Run: python -m rag_platform.cli run configs/experiments/baseline_full_wikipedia_ragas.yaml")
        print("3. Check results in the ./results directory")


if __name__ == "__main__":
    main()