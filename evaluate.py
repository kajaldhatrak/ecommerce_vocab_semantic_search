"""
Evaluation Framework Module
===========================

This module implements evaluation metrics for comparing search methods:
- Precision@k: How many retrieved results are relevant
- Recall@k: How many relevant items were retrieved
- Mean Reciprocal Rank (MRR): Average position of first relevant result
- nDCG@k: Normalized Discounted Cumulative Gain

Understanding These Metrics:
----------------------------

Precision@k = |{relevant docs in top k}| / k
- Measures how many of the returned results are actually relevant
- High precision = few irrelevant results shown
- Important for user experience (no junk results)

Recall@k = |{relevant docs in top k}| / |{all relevant docs}|
- Measures how many relevant items were found
- High recall = good coverage of relevant items
- Important when users want comprehensive results

MRR = (1/|Q|) * Σ (1/rank_i)
- Average of reciprocal ranks across queries
- Emphasizes first relevant result position
- High MRR = relevant results appear early

nDCG@k = DCG@k / IDCG@k
- Normalized Discounted Cumulative Gain
- DCG@k = Σ rel_i / log2(i+1) for i=1 to k
- Accounts for graded relevance and position
- Perfect ranking = 1.0, worst = 0.0

Why These Matter for E-commerce:
- Users rarely look past first page (Precision crucial)
- Finding THE product matters more than finding ALL (MRR)
- Position matters: top 3 results are 10x more valuable (nDCG)
"""

import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Precision@k.
    
    Precision@k = |{relevant ∩ retrieved[:k]}| / k
    
    Measures the fraction of retrieved documents that are relevant.
    
    Args:
        retrieved: List of retrieved document IDs (in order)
        relevant: List of relevant document IDs (ground truth)
        k: Number of top results to consider
    
    Returns:
        Precision score between 0 and 1
    """
    if k == 0:
        return 0.0
    
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)
    
    relevant_retrieved = retrieved_at_k & relevant_set
    
    return len(relevant_retrieved) / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Recall@k.
    
    Recall@k = |{relevant ∩ retrieved[:k]}| / |{relevant}|
    
    Measures the fraction of relevant documents that were retrieved.
    
    Args:
        retrieved: List of retrieved document IDs (in order)
        relevant: List of relevant document IDs (ground truth)
        k: Number of top results to consider
    
    Returns:
        Recall score between 0 and 1
    """
    if len(relevant) == 0:
        return 0.0
    
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)
    
    relevant_retrieved = retrieved_at_k & relevant_set
    
    return len(relevant_retrieved) / len(relevant_set)


def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    RR = 1/rank of first relevant document
    MRR = average RR across queries
    
    This function calculates RR for a single query.
    Emphasizes the position of the FIRST relevant result.
    
    Args:
        retrieved: List of retrieved document IDs (in order)
        relevant: List of relevant document IDs (ground truth)
    
    Returns:
        Reciprocal rank score (1/position of first relevant)
    """
    relevant_set = set(relevant)
    
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant_set:
            return 1.0 / rank
    
    return 0.0


def dcg_at_k(relevances: List[float], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at k.
    
    DCG@k = Σ (2^rel_i - 1) / log2(i + 1) for i = 1 to k
    
    Modified formula using 2^rel for stronger emphasis on highly relevant items.
    
    Args:
        relevances: List of relevance scores for each position
        k: Cutoff position
    
    Returns:
        DCG score
    """
    relevances = relevances[:k]
    
    if len(relevances) == 0:
        return 0.0
    
    dcg = relevances[0]  # Position 1
    
    for i in range(1, len(relevances)):
        dcg += relevances[i] / np.log2(i + 2)  # log2(position + 1)
    
    return dcg


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.
    
    nDCG@k = DCG@k / IDCG@k
    
    Where IDCG is the ideal DCG (perfect ranking).
    
    For binary relevance (relevant or not), we use:
    - rel = 1 for relevant documents
    - rel = 0 for non-relevant documents
    
    Why nDCG matters:
    - Considers position: Higher positions valued more
    - Normalized: 1.0 means perfect ranking
    - Works with graded relevance (though we use binary here)
    
    Args:
        retrieved: List of retrieved document IDs (in order)
        relevant: List of relevant document IDs (ground truth)
        k: Cutoff position
    
    Returns:
        nDCG score between 0 and 1
    """
    relevant_set = set(relevant)
    
    # Calculate relevance scores for retrieved documents
    retrieved_relevances = [1.0 if doc_id in relevant_set else 0.0 for doc_id in retrieved[:k]]
    
    # Calculate DCG
    dcg = dcg_at_k(retrieved_relevances, k)
    
    # Calculate ideal DCG (all relevant docs at top)
    ideal_relevances = [1.0] * min(len(relevant_set), k)
    idcg = dcg_at_k(ideal_relevances, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def evaluate_single_query(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int = 10
) -> Dict[str, float]:
    """
    Evaluate a single query across all metrics.
    
    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of ground truth relevant IDs
        k: Cutoff for @k metrics
    
    Returns:
        Dictionary of metric names to scores
    """
    return {
        f"precision@{k}": precision_at_k(retrieved_ids, relevant_ids, k),
        f"recall@{k}": recall_at_k(retrieved_ids, relevant_ids, k),
        "mrr": mean_reciprocal_rank(retrieved_ids, relevant_ids),
        f"ndcg@{k}": ndcg_at_k(retrieved_ids, relevant_ids, k)
    }


class SearchEvaluator:
    """
    Comprehensive evaluation framework for search methods.
    
    Runs evaluation across all queries and computes aggregate metrics.
    """
    
    def __init__(
        self,
        search_engine,
        queries: List[Dict[str, Any]],
        k: int = 10
    ):
        """
        Initialize evaluator.
        
        Args:
            search_engine: SearchEngine instance
            queries: List of query dicts with 'query' and 'relevant_product_ids'
            k: Cutoff for @k metrics
        """
        self.engine = search_engine
        self.queries = queries
        self.k = k
        self.results = {}
    
    def evaluate_method(
        self,
        method: str,
        show_progress: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate a single search method across all queries.
        
        Args:
            method: One of 'bm25', 'vector', 'hybrid'
            show_progress: Show progress bar
        
        Returns:
            Aggregated metrics dictionary
        """
        all_metrics = defaultdict(list)
        
        iterator = tqdm(self.queries, desc=f"Evaluating {method}") if show_progress else self.queries
        
        for query_data in iterator:
            query_text = query_data["query"]
            relevant_ids = query_data["relevant_product_ids"]
            
            # Run search
            if method == "bm25":
                results = self.engine.search_keyword(query_text, top_k=self.k)
            elif method == "vector":
                results = self.engine.search_vector(query_text, top_k=self.k)
            elif method == "hybrid":
                results = self.engine.search_hybrid(query_text, top_k=self.k)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Get retrieved IDs
            retrieved_ids = [r["id"] for r in results]
            
            # Calculate metrics
            metrics = evaluate_single_query(retrieved_ids, relevant_ids, self.k)
            
            for metric_name, score in metrics.items():
                all_metrics[metric_name].append(score)
        
        # Aggregate metrics (mean)
        aggregated = {}
        for metric_name, scores in all_metrics.items():
            aggregated[metric_name] = np.mean(scores)
        
        self.results[method] = aggregated
        return aggregated
    
    def evaluate_all_methods(
        self,
        show_progress: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all three search methods.
        
        Returns:
            Dictionary with results for each method
        """
        logger.info("=" * 60)
        logger.info("Evaluating Search Methods")
        logger.info("=" * 60)
        
        methods = ["bm25", "vector", "hybrid"]
        
        for method in methods:
            self.evaluate_method(method, show_progress)
        
        return self.results
    
    def evaluate_by_query_type(
        self,
        show_progress: bool = True
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Evaluate methods grouped by query type.
        
        This shows how different search methods perform on:
        - Synonym queries: Vector should excel
        - Keyword queries: BM25 should be competitive
        - Natural language: Vector should excel
        - Multi-concept: Hybrid should be best
        
        Returns:
            Nested dict: query_type -> method -> metrics
        """
        # Group queries by type
        queries_by_type = defaultdict(list)
        for query in self.queries:
            query_type = query.get("query_type", "unknown")
            queries_by_type[query_type].append(query)
        
        results_by_type = {}
        
        for query_type, type_queries in queries_by_type.items():
            logger.info(f"\nEvaluating query type: {query_type} ({len(type_queries)} queries)")
            
            # Temporarily swap queries
            original_queries = self.queries
            self.queries = type_queries
            self.results = {}
            
            # Evaluate
            type_results = self.evaluate_all_methods(show_progress=False)
            results_by_type[query_type] = type_results.copy()
            
            self.queries = original_queries
        
        self.results = {}  # Reset
        return results_by_type
    
    def print_results(self):
        """Print evaluation results in a formatted table."""
        if not self.results:
            logger.warning("No results to print. Run evaluate_all_methods first.")
            return
        
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        
        # Header
        metrics = list(list(self.results.values())[0].keys())
        header = f"{'Method':<12}" + "".join(f"{m:>15s}" for m in metrics)
        print(header)
        print("-" * len(header))
        
        # Rows
        for method, method_metrics in self.results.items():
            row = f"{method:<12}"
            for metric in metrics:
                score = method_metrics[metric]
                row += f"{score:>15.4f}"
            print(row)
        
        print("=" * 80)
    
    def compute_improvements(self) -> Dict[str, Dict[str, float]]:
        """
        Compute percentage improvement of vector and hybrid over BM25.
        
        Returns:
            Dictionary with improvement percentages
        """
        if "bm25" not in self.results:
            return {}
        
        bm25_metrics = self.results["bm25"]
        improvements = {}
        
        for method in ["vector", "hybrid"]:
            if method not in self.results:
                continue
            
            method_metrics = self.results[method]
            improvements[method] = {}
            
            for metric, bm25_score in bm25_metrics.items():
                if bm25_score > 0:
                    improvement = ((method_metrics[metric] - bm25_score) / bm25_score) * 100
                else:
                    improvement = 0 if method_metrics[metric] == 0 else float('inf')
                
                improvements[method][metric] = improvement
        
        return improvements
    
    def print_improvements(self):
        """Print improvement percentages."""
        improvements = self.compute_improvements()
        
        if not improvements:
            return
        
        print("\n" + "=" * 80)
        print("IMPROVEMENT OVER BM25 (%)")
        print("=" * 80)
        
        metrics = list(list(self.results.values())[0].keys())
        header = f"{'Method':<12}" + "".join(f"{m:>15s}" for m in metrics)
        print(header)
        print("-" * len(header))
        
        for method, method_improvements in improvements.items():
            row = f"{method:<12}"
            for metric in metrics:
                score = method_improvements[metric]
                sign = "+" if score >= 0 else ""
                row += f"{sign}{score:>13.1f}%"
            print(row)
        
        print("=" * 80)


def generate_visualization(
    results: Dict[str, Dict[str, float]],
    output_path: str = "search_comparison.png"
):
    """
    Generate bar chart comparing search methods.
    
    Args:
        results: Evaluation results for each method
        output_path: Path to save the PNG
    """
    import matplotlib.pyplot as plt
    
    methods = list(results.keys())
    metrics = list(results[methods[0]].keys())
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.25
    
    colors = {'bm25': '#2196F3', 'vector': '#4CAF50', 'hybrid': '#FF9800'}
    
    for i, method in enumerate(methods):
        scores = [results[method][m] for m in metrics]
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=method.upper(), color=colors.get(method, '#999999'))
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.annotate(f'{score:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Search Method Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('@', '@') for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to {output_path}")


def generate_query_type_visualization(
    results_by_type: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str = "search_by_query_type.png"
):
    """
    Generate visualization showing performance by query type.
    
    This demonstrates that:
    - BM25 works well for keyword queries
    - Vector excels on synonym and natural language queries
    - Hybrid provides consistent performance across all types
    
    Args:
        results_by_type: Results grouped by query type
        output_path: Path to save the PNG
    """
    import matplotlib.pyplot as plt
    
    query_types = list(results_by_type.keys())
    methods = ['bm25', 'vector', 'hybrid']
    
    # Use MRR as the representative metric
    metric = 'mrr'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(query_types))
    width = 0.25
    
    colors = {'bm25': '#2196F3', 'vector': '#4CAF50', 'hybrid': '#FF9800'}
    
    for i, method in enumerate(methods):
        scores = []
        for qt in query_types:
            if method in results_by_type[qt]:
                scores.append(results_by_type[qt][method].get(metric, 0))
            else:
                scores.append(0)
        
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, scores, width, label=method.upper(), color=colors.get(method, '#999999'))
    
    ax.set_xlabel('Query Type')
    ax.set_ylabel(f'{metric.upper()} Score')
    ax.set_title('Search Performance by Query Type')
    ax.set_xticks(x)
    ax.set_xticklabels(query_types, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to {output_path}")


def load_queries(filepath: str = "queries.json") -> List[Dict[str, Any]]:
    """Load queries from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """Main function to run evaluation."""
    from index_setup import get_elasticsearch_client
    from search import SearchEngine
    
    logger.info("=" * 60)
    logger.info("E-commerce Semantic Search - Evaluation")
    logger.info("=" * 60)
    
    # Load queries
    queries = load_queries("queries.json")
    logger.info(f"Loaded {len(queries)} queries")
    
    # Connect to Elasticsearch
    client = get_elasticsearch_client()
    
    # Create search engine
    engine = SearchEngine(client)
    
    # Create evaluator
    evaluator = SearchEvaluator(engine, queries, k=10)
    
    # Run evaluation
    results = evaluator.evaluate_all_methods()
    
    # Print results
    evaluator.print_results()
    evaluator.print_improvements()
    
    # Generate visualization
    generate_visualization(results)
    
    # Evaluate by query type
    logger.info("\nEvaluating by query type...")
    results_by_type = evaluator.evaluate_by_query_type()
    
    print("\n" + "=" * 80)
    print("RESULTS BY QUERY TYPE")
    print("=" * 80)
    
    for query_type, type_results in results_by_type.items():
        print(f"\n{query_type.upper()}:")
        for method, metrics in type_results.items():
            print(f"  {method:8s}: MRR={metrics['mrr']:.3f}, P@10={metrics['precision@10']:.3f}, R@10={metrics['recall@10']:.3f}")
    
    # Generate query type visualization
    generate_query_type_visualization(results_by_type)
    
    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
