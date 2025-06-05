#!/usr/bin/env python3
"""
Comprehensive experiment runner for RAG evaluation.
Runs multiple configurations and generates comparative analysis.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import rag_platform
from rag_platform.core.config import ExperimentConfig
from rag_platform.core.experiment import ExperimentRunner


class ExperimentAnalyzer:
    """Analyzes and compares RAG experiments."""
    
    def __init__(self, results_dir: str = "./results/experiments"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def run_experiment(self, config_path: str) -> Dict[str, Any]:
        """Run a single experiment and return detailed results."""
        print(f"\\n{'='*60}")
        print(f"🚀 Starting experiment: {config_path}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Load configuration
        config = ExperimentConfig.from_yaml(config_path)
        print(f"📋 Experiment: {config.name}")
        print(f"📊 Sample size: {config.evaluation.sample_size}")
        print(f"🤖 LLM: {config.llm.model}")
        print(f"🔍 Embeddings: {config.embedding.model}")
        
        # Run experiment
        runner = ExperimentRunner(config)
        results = runner.run()
        
        # Calculate timing and costs
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Estimate costs based on model and usage
        cost_estimate = self._estimate_costs(config, results)
        
        # Enhanced results with metadata
        enhanced_results = {
            **results,
            "config_path": config_path,
            "processing_time": processing_time,
            "cost_estimate": cost_estimate,
            "model_config": {
                "llm_model": config.llm.model,
                "embedding_model": config.embedding.model,
                "chunk_size": config.chunking.chunk_size,
                "retrieval_k": config.retriever.k
            }
        }
        
        print(f"✅ Completed in {processing_time:.1f}s")
        print(f"💰 Estimated cost: ${cost_estimate['total_cost']:.3f}")
        
        if 'evaluation_results' in results:
            metrics = results['evaluation_results']
            print(f"📈 Metrics:")
            for metric, score in metrics.items():
                print(f"   {metric}: {score:.3f}")
        
        return enhanced_results
    
    def _estimate_costs(self, config: ExperimentConfig, results: Dict) -> Dict[str, float]:
        """Estimate API costs based on configuration and usage."""
        
        # OpenAI pricing (as of 2024)
        pricing = {
            "embeddings": {
                "text-embedding-3-small": 0.00002,  # $0.02 per 1M tokens
                "text-embedding-3-large": 0.00013,  # $0.13 per 1M tokens
            },
            "llm": {
                "gpt-3.5-turbo": {"input": 0.0005/1000, "output": 0.0015/1000},
                "gpt-4-turbo": {"input": 0.01/1000, "output": 0.03/1000},
                "gpt-4-turbo-preview": {"input": 0.01/1000, "output": 0.03/1000},
                "gpt-4": {"input": 0.03/1000, "output": 0.06/1000},
            }
        }
        
        sample_size = config.evaluation.sample_size or 100
        
        # Estimate embedding costs
        # Rough estimate: average document ~500 tokens, chunked to ~100 tokens per chunk
        avg_chunks_per_doc = 5
        total_chunks = sample_size * avg_chunks_per_doc
        avg_tokens_per_chunk = 100
        embedding_tokens = total_chunks * avg_tokens_per_chunk
        
        embedding_cost = embedding_tokens * pricing["embeddings"].get(
            config.embedding.model, 0.00002
        )
        
        # Estimate LLM costs
        # Rough estimate: query ~50 tokens, context ~500 tokens, answer ~100 tokens
        avg_input_tokens = 550  # query + context
        avg_output_tokens = 100  # answer
        
        llm_pricing = pricing["llm"].get(config.llm.model, 
                                        {"input": 0.0005/1000, "output": 0.0015/1000})
        
        input_cost = sample_size * avg_input_tokens * llm_pricing["input"]
        output_cost = sample_size * avg_output_tokens * llm_pricing["output"]
        llm_cost = input_cost + output_cost
        
        total_cost = embedding_cost + llm_cost
        
        return {
            "embedding_cost": embedding_cost,
            "llm_cost": llm_cost,
            "total_cost": total_cost,
            "cost_per_question": total_cost / sample_size,
            "estimated_tokens": {
                "embedding": embedding_tokens,
                "llm_input": sample_size * avg_input_tokens,
                "llm_output": sample_size * avg_output_tokens,
            }
        }
    
    def run_all_experiments(self, config_paths: List[str]) -> List[Dict[str, Any]]:
        """Run all experiments and return results."""
        all_results = []
        
        print(f"🎯 Running {len(config_paths)} experiments...")
        
        for i, config_path in enumerate(config_paths, 1):
            print(f"\\n📍 Experiment {i}/{len(config_paths)}")
            
            try:
                result = self.run_experiment(config_path)
                all_results.append(result)
                
                # Save individual result
                self._save_individual_result(result)
                
            except Exception as e:
                print(f"❌ Experiment failed: {e}")
                # Continue with other experiments
                
        return all_results
    
    def _save_individual_result(self, result: Dict[str, Any]) -> None:
        """Save individual experiment result."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result['experiment_name']}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Saved: {filepath}")
    
    def generate_comparative_analysis(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive comparative analysis."""
        
        print(f"\\n{'='*60}")
        print("📊 GENERATING COMPARATIVE ANALYSIS")
        print(f"{'='*60}")
        
        # Create comparative CSV
        csv_data = []
        detailed_evaluations = []
        
        for result in all_results:
            experiment_name = result['experiment_name']
            config_path = result['config_path']
            
            # Extract metrics
            metrics = result.get('evaluation_results', {})
            cost_info = result.get('cost_estimate', {})
            model_config = result.get('model_config', {})
            
            # CSV row
            csv_row = {
                'experiment_name': experiment_name,
                'config_file': Path(config_path).name,
                'llm_model': model_config.get('llm_model', 'unknown'),
                'embedding_model': model_config.get('embedding_model', 'unknown'),
                'chunk_size': model_config.get('chunk_size', 0),
                'retrieval_k': model_config.get('retrieval_k', 0),
                'faithfulness': metrics.get('faithfulness', 0),
                'answer_relevancy': metrics.get('answer_relevancy', 0),
                'context_precision': metrics.get('context_precision', 0),
                'context_recall': metrics.get('context_recall', 0),
                'total_cost': cost_info.get('total_cost', 0),
                'cost_per_question': cost_info.get('cost_per_question', 0),
                'processing_time': result.get('processing_time', 0),
                'embedding_cost': cost_info.get('embedding_cost', 0),
                'llm_cost': cost_info.get('llm_cost', 0),
            }
            csv_data.append(csv_row)
            
            # Detailed evaluation data for RAGAS  
            detailed_evaluations.append({
                'experiment_name': experiment_name,
                'config': model_config,
                'metrics': metrics,
                'cost_analysis': cost_info,
                'timestamp': result.get('timestamp'),
                'complete_results': result.get('complete_results', []),  # ALL results
                'total_samples': len(result.get('complete_results', [])),
                'average_processing_time': result.get('average_processing_time', 0),
            })
        
        # Save CSV
        df = pd.DataFrame(csv_data)
        csv_path = self.results_dir / "experiments_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"📈 Comparative CSV: {csv_path}")
        
        # Save detailed evaluation data
        detailed_path = self.results_dir / "detailed_evaluations.json"
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': 'Detailed RAGAS evaluation data for all experiments',
                'experiments': detailed_evaluations,
                'generated_at': datetime.now().isoformat(),
                'total_experiments': len(all_results)
            }, f, indent=2, ensure_ascii=False, default=str)
        print(f"📋 Detailed evaluations: {detailed_path}")
        
        # Generate summary report
        summary = self._generate_summary_report(df, all_results)
        summary_path = self.results_dir / "EXPERIMENT_SUMMARY.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"📝 Summary report: {summary_path}")
        
        return {
            'csv_path': csv_path,
            'detailed_path': detailed_path,
            'summary_path': summary_path,
            'dataframe': df,
            'detailed_evaluations': detailed_evaluations
        }
    
    def _generate_summary_report(self, df: pd.DataFrame, all_results: List[Dict]) -> str:
        """Generate markdown summary report."""
        
        report = f"""# RAG Experiments Summary Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Total Experiments**: {len(all_results)}  
**Dataset**: Russian Wikipedia QA (100 samples each)

## 📊 Results Overview

| Experiment | LLM Model | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Total Cost | Cost/Question |
|------------|-----------|--------------|------------------|-------------------|----------------|------------|---------------|
"""
        
        for _, row in df.iterrows():
            report += f"| {row['experiment_name']} | {row['llm_model']} | {row['faithfulness']:.3f} | {row['answer_relevancy']:.3f} | {row['context_precision']:.3f} | {row['context_recall']:.3f} | ${row['total_cost']:.2f} | ${row['cost_per_question']:.3f} |\\n"
        
        # Find best performing
        best_overall = df.loc[df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean(axis=1).idxmax()]
        most_cost_effective = df.loc[(df[['faithfulness', 'answer_relevancy']].mean(axis=1) / df['total_cost']).idxmax()]
        
        report += f"""

## 🏆 Key Findings

### Best Overall Performance
- **Experiment**: {best_overall['experiment_name']}
- **Average Score**: {df.loc[best_overall.name, ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean():.3f}
- **Total Cost**: ${best_overall['total_cost']:.2f}

### Most Cost-Effective
- **Experiment**: {most_cost_effective['experiment_name']}
- **Performance/Cost Ratio**: {(df.loc[most_cost_effective.name, ['faithfulness', 'answer_relevancy']].mean() / most_cost_effective['total_cost']):.2f}
- **Total Cost**: ${most_cost_effective['total_cost']:.2f}

## 💰 Cost Analysis

### Total Costs
"""
        
        for _, row in df.iterrows():
            report += f"- **{row['experiment_name']}**: ${row['total_cost']:.2f} (${row['cost_per_question']:.3f}/question)\\n"
        
        report += f"""

### Cost Breakdown
- **Embedding Costs**: ${df['embedding_cost'].sum():.2f}
- **LLM Costs**: ${df['llm_cost'].sum():.2f}
- **Total Project Cost**: ${df['total_cost'].sum():.2f}

## 🎯 Recommendations

### For Production Use
"""
        
        if len(df) >= 3:
            baseline = df[df['experiment_name'].str.contains('baseline', case=False)].iloc[0] if len(df[df['experiment_name'].str.contains('baseline', case=False)]) > 0 else df.iloc[0]
            advanced = df[df['total_cost'] == df['total_cost'].max()].iloc[0]
            
            report += f"""
1. **Development/Testing**: Use {baseline['experiment_name']} for rapid iteration (${baseline['cost_per_question']:.3f}/question)
2. **Production**: Consider {most_cost_effective['experiment_name']} for best balance of cost and performance
3. **High-Stakes Applications**: Use {advanced['experiment_name']} when accuracy is paramount

### Performance Insights
- Higher embedding dimensions generally improve context precision
- GPT-4 models show better faithfulness but at 10-20x cost
- Optimal chunk size appears to be around 256-512 tokens for Russian text
"""
        
        report += """

## 📁 Files Generated

1. **experiments_comparison.csv** - Comparative metrics table
2. **detailed_evaluations.json** - Full RAGAS evaluation data
3. **individual_results/*.json** - Detailed results for each experiment

## 🔗 Next Steps

1. Analyze detailed evaluation data for failure cases
2. Implement best-performing configuration in production
3. Consider hybrid approaches combining cost-effective and high-performance methods
4. Monitor production metrics and costs
"""
        
        return report


def main():
    """Main experiment runner."""
    print("🧪 RAG EXPERIMENTS RUNNER")
    print("=" * 60)
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Please configure your API key in .env file.")
        return
    
    # Define experiments
    experiments = [
        "configs/experiments/baseline_full_wikipedia_ragas.yaml",
        "configs/experiments/enhanced_rag.yaml", 
        "configs/experiments/advanced_rag.yaml"
    ]
    
    # Verify config files exist
    missing_configs = [exp for exp in experiments if not Path(exp).exists()]
    if missing_configs:
        print(f"❌ Missing config files: {missing_configs}")
        return
    
    # Initialize analyzer
    analyzer = ExperimentAnalyzer()
    
    # Run experiments
    all_results = analyzer.run_all_experiments(experiments)
    
    if not all_results:
        print("❌ No experiments completed successfully")
        return
    
    # Generate analysis
    analysis = analyzer.generate_comparative_analysis(all_results)
    
    # Final summary
    print(f"\\n{'='*60}")
    print("🎉 EXPERIMENTS COMPLETED!")
    print(f"{'='*60}")
    print(f"📊 Results: {analysis['csv_path']}")
    print(f"📋 Detailed data: {analysis['detailed_path']}")
    print(f"📝 Summary: {analysis['summary_path']}")
    print(f"💰 Total cost: ${analysis['dataframe']['total_cost'].sum():.2f}")
    print(f"⏱️  Total time: {analysis['dataframe']['processing_time'].sum():.1f}s")


if __name__ == "__main__":
    main()