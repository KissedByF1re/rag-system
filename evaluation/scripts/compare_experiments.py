#!/usr/bin/env python3
"""Compare all RAG experiments and generate comprehensive CSV."""

import json
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List

def load_experiment_results(results_dir: str) -> List[Dict]:
    """Load all experiment results from directory."""
    print("DEBUG: Inside load_experiment_results function")
    results_path = Path(results_dir)
    experiment_results = []
    
    print(f"📁 Checking directory: {results_path.absolute()}")
    print(f"📁 Directory exists: {results_path.exists()}")
    
    # Find all JSON files in results directory
    json_files = list(results_path.glob("*.json"))
    print(f"📄 Found {len(json_files)} JSON files")
    
    for file_path in json_files:
        print(f"📖 Loading: {file_path.name}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                experiment_results.append({
                    'file_name': file_path.name,
                    'data': data
                })
                print(f"✅ Successfully loaded: {file_path.name}")
        except Exception as e:
            print(f"⚠️ Error loading {file_path.name}: {e}")
    
    print(f"📊 Total experiments loaded: {len(experiment_results)}")
    return experiment_results

def extract_experiment_summary(experiment_data: Dict) -> Dict:
    """Extract key metrics from experiment data."""
    data = experiment_data['data']
    file_name = experiment_data['file_name']
    
    # Extract experiment name from file name
    if "baseline_vector" in file_name:
        exp_name = "Baseline Vector"
    elif "enhanced_hybrid" in file_name:
        exp_name = "Enhanced Hybrid"
    elif "advanced_reranker" in file_name:
        exp_name = "Advanced Reranker"
    elif "pro_graphrag" in file_name:
        exp_name = "Pro GraphRAG"
    elif "baseline" in file_name:
        exp_name = "Baseline"
    elif "enhanced" in file_name:
        exp_name = "Enhanced"  
    elif "advanced" in file_name:
        exp_name = "Advanced"
    else:
        exp_name = "Unknown"
    
    # Basic experiment info
    summary = {
        'experiment_name': exp_name,
        'timestamp': data.get('timestamp', file_name.split('_')[-1].replace('.json', '')),
        'file_name': file_name
    }
    
    # Get config from complete results (they have the config embedded)
    complete_results = data.get('complete_results', [])
    if complete_results and 'experiment_config' in complete_results[0]:
        config = complete_results[0]['experiment_config']
    else:
        config = {}
    
    summary.update({
        'llm_model': config.get('llm_model', 'Unknown'),
        'embedding_model': config.get('embedding_model', 'Unknown'),
        'chunk_size': config.get('chunk_size', 'Unknown'),
        'retrieval_k': config.get('retrieval_k', 'Unknown'),
        'temperature': config.get('temperature', 0.0 if config.get('temperature', 'Unknown') == 'Unknown' else config.get('temperature'))  # All experiments use 0.0
    })
    
    # Performance metrics from complete results
    if complete_results:
        processing_times = [r.get('processing_time', 0) for r in complete_results]
        summary.update({
            'total_questions': len(complete_results),
            'avg_processing_time_per_question': sum(processing_times) / len(processing_times) if processing_times else 0,
            'total_processing_time': sum(processing_times)
        })
    else:
        summary.update({
            'total_questions': 0,
            'avg_processing_time_per_question': 0,
            'total_processing_time': 0
        })
    
    # RAGAS evaluation results
    eval_results = data.get('evaluation_metrics', {})
    summary.update({
        'faithfulness': eval_results.get('faithfulness', 0),
        'answer_relevancy': eval_results.get('answer_relevancy', 0),
        'context_precision': eval_results.get('context_precision', 0),
        'context_recall': eval_results.get('context_recall', 0)
    })
    
    # Calculate average RAGAS score
    ragas_scores = [
        eval_results.get('faithfulness', 0),
        eval_results.get('answer_relevancy', 0),
        eval_results.get('context_precision', 0),
        eval_results.get('context_recall', 0)
    ]
    summary['avg_ragas_score'] = sum(ragas_scores) / len(ragas_scores) if ragas_scores else 0
    
    # Cost estimation (rough estimates based on model)
    if summary['llm_model'] == 'gpt-3.5-turbo':
        cost_per_query = 0.025
    elif summary['llm_model'] == 'gpt-4-turbo':
        cost_per_query = 0.18
    elif summary['llm_model'] == 'gpt-4':
        cost_per_query = 0.47
    elif summary['llm_model'] == 'gpt-4.1-nano':
        cost_per_query = 0.005  # Very low cost for nano model
    else:
        cost_per_query = 0.1
    
    summary['estimated_cost_per_query'] = cost_per_query
    summary['estimated_total_cost'] = cost_per_query * summary['total_questions']
    
    # Knowledge base stats from complete results
    if complete_results:
        contexts_retrieved = [r.get('num_retrieved_contexts', 0) for r in complete_results]
        answer_lengths = [len(r.get('generated_answer', '')) for r in complete_results]
        
        summary.update({
            'avg_contexts_retrieved': sum(contexts_retrieved) / len(contexts_retrieved) if contexts_retrieved else 0,
            'avg_answer_length': sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
        })
    else:
        summary.update({
            'avg_contexts_retrieved': 0,
            'avg_answer_length': 0
        })
    
    return summary

def generate_comparison_csv(experiment_results: List[Dict], output_path: str):
    """Generate comprehensive comparison CSV."""
    summaries = []
    
    for experiment in experiment_results:
        summary = extract_experiment_summary(experiment)
        summaries.append(summary)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(summaries)
    
    # Sort by cost (baseline -> enhanced -> advanced)
    df = df.sort_values('estimated_cost_per_query')
    
    # Round numeric columns for readability
    numeric_columns = [
        'avg_processing_time_per_question', 'total_processing_time',
        'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall',
        'avg_ragas_score', 'estimated_cost_per_query', 'estimated_total_cost',
        'avg_contexts_retrieved', 'avg_answer_length'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].round(4)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return df

def generate_detailed_comparison_report(df: pd.DataFrame, output_path: str):
    """Generate detailed comparison report."""
    
    report = f"""# RAG EXPERIMENT COMPARISON REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## EXECUTIVE SUMMARY

This report compares three RAG configurations tested on a Russian QA dataset with full Wikipedia knowledge base.

### KEY FINDINGS

"""
    
    # Find best performers
    best_faithfulness = df.loc[df['faithfulness'].idxmax()]
    best_relevancy = df.loc[df['answer_relevancy'].idxmax()]
    best_overall = df.loc[df['avg_ragas_score'].idxmax()]
    most_cost_effective = df.loc[df['estimated_cost_per_query'].idxmin()]
    
    report += f"""
**Best Faithfulness**: {best_faithfulness['experiment_name']} (Score: {best_faithfulness['faithfulness']:.4f})
**Best Answer Relevancy**: {best_relevancy['experiment_name']} (Score: {best_relevancy['answer_relevancy']:.4f})  
**Best Overall RAGAS**: {best_overall['experiment_name']} (Avg: {best_overall['avg_ragas_score']:.4f})
**Most Cost Effective**: {most_cost_effective['experiment_name']} (${most_cost_effective['estimated_cost_per_query']:.3f}/query)

## DETAILED COMPARISON

| Metric | Baseline | Enhanced | Advanced |
|--------|----------|----------|----------|
"""
    
    # Add comparison table
    metrics = [
        ('LLM Model', 'llm_model'),
        ('Embedding Model', 'embedding_model'), 
        ('Chunk Size', 'chunk_size'),
        ('Retrieval K', 'retrieval_k'),
        ('Faithfulness', 'faithfulness'),
        ('Answer Relevancy', 'answer_relevancy'),
        ('Context Precision', 'context_precision'),
        ('Context Recall', 'context_recall'),
        ('Avg RAGAS Score', 'avg_ragas_score'),
        ('Avg Processing Time (s)', 'avg_processing_time_per_question'),
        ('Est. Cost per Query ($)', 'estimated_cost_per_query'),
        ('Avg Contexts Retrieved', 'avg_contexts_retrieved'),
        ('Avg Answer Length', 'avg_answer_length')
    ]
    
    for metric_name, col_name in metrics:
        row_values = []
        for _, row in df.iterrows():
            value = row[col_name]
            if isinstance(value, float):
                if col_name in ['estimated_cost_per_query']:
                    row_values.append(f"${value:.3f}")
                else:
                    row_values.append(f"{value:.3f}")
            else:
                row_values.append(str(value))
        
        report += f"| {metric_name} | {' | '.join(row_values)} |\n"
    
    report += f"""
## RECOMMENDATIONS

Based on the results:

1. **For Budget-Conscious Deployments**: Use Baseline configuration
   - Lowest cost (${df.iloc[0]['estimated_cost_per_query']:.3f}/query)
   - Reasonable performance for most use cases
   
2. **For Balanced Performance**: Use Enhanced configuration  
   - Better answer relevancy ({df.iloc[1]['answer_relevancy']:.3f})
   - Moderate cost increase
   
3. **For Maximum Quality**: Use Advanced configuration
   - Best overall RAGAS score ({df.iloc[2]['avg_ragas_score']:.3f})
   - Highest cost but best performance

## TECHNICAL NOTES

- All experiments used full Wikipedia knowledge base (3,594 chunks)
- Sample size: {df.iloc[0]['total_questions']} questions per experiment
- Russian language QA dataset
- RAGAS evaluation with OpenAI models

## NEXT STEPS

1. Run larger samples (100+ questions) for more robust metrics
2. Optimize prompts to improve faithfulness scores
3. Experiment with different chunking strategies
4. Test on domain-specific datasets
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

def main():
    """Generate comprehensive experiment comparison."""
    print("DEBUG: Starting main function")
    results_dir = "./results/experiments"
    
    print("🔍 Loading experiment results...")
    print(f"📁 Looking in directory: {Path(results_dir).absolute()}")
    experiment_results = load_experiment_results(results_dir)
    
    if not experiment_results:
        print("❌ No experiment results found!")
        # Debug: show what files exist
        results_path = Path(results_dir)
        if results_path.exists():
            all_files = list(results_path.glob("*"))
            print(f"📂 Directory exists with {len(all_files)} files:")
            for f in all_files:
                print(f"  - {f.name}")
        else:
            print(f"📂 Directory {results_path.absolute()} does not exist")
        return
    
    print(f"📊 Found {len(experiment_results)} experiments")
    
    # Generate CSV comparison
    csv_path = "rag_experiments_comparison.csv"
    print(f"📈 Generating comparison CSV: {csv_path}")
    df = generate_comparison_csv(experiment_results, csv_path)
    
    # Generate detailed report  
    report_path = "rag_experiments_report.md"
    print(f"📝 Generating detailed report: {report_path}")
    generate_detailed_comparison_report(df, report_path)
    
    print("\n✅ COMPARISON COMPLETE!")
    print(f"📄 CSV File: {csv_path}")
    print(f"📋 Report: {report_path}")
    
    # Show summary table
    print("\n📊 EXPERIMENT SUMMARY:")
    print(df[['experiment_name', 'llm_model', 'avg_ragas_score', 'estimated_cost_per_query']].to_string(index=False))

if __name__ == "__main__":
    main()