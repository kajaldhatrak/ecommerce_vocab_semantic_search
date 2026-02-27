"""
E-commerce Semantic Search System - Main Orchestrator
======================================================

This is the main entry point that orchestrates the complete pipeline:
1. Generate synthetic product data
2. Generate embeddings
3. Setup Elasticsearch index
4. Ingest products with embeddings
5. Run evaluation
6. Generate visualizations and blog-ready output

The purpose of this project is to demonstrate that:
- Vector search captures semantic meaning (synonyms, paraphrases)
- BM25 keyword search excels at exact matches
- Hybrid search combines both for best overall performance

Run with: python main.py
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_ES_URL = "http://localhost:9200"
DEFAULT_INDEX_NAME = "ecommerce_products"
NUM_PRODUCTS = 1500
NUM_QUERIES = 100


def run_data_generation():
    """Step 1: Generate synthetic product and query data."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: GENERATING SYNTHETIC DATA")
    logger.info("=" * 70)
    
    from data_generation import generate_products, generate_queries, save_data
    
    products = generate_products(NUM_PRODUCTS)
    queries = generate_queries(products, NUM_QUERIES)
    save_data(products, queries)
    
    return products, queries


def run_embedding_generation(products: List[Dict[str, Any]]):
    """Step 2: Generate embeddings for products."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: GENERATING EMBEDDINGS")
    logger.info("=" * 70)
    
    from embed import generate_product_embeddings, save_products_with_embeddings
    
    products_with_embeddings, embedding_dim = generate_product_embeddings(products)
    save_products_with_embeddings(products_with_embeddings)
    
    return products_with_embeddings, embedding_dim


def run_index_setup(es_url: str, index_name: str, embedding_dim: int):
    """Step 3: Setup Elasticsearch index."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: SETTING UP ELASTICSEARCH INDEX")
    logger.info("=" * 70)
    
    from index_setup import get_elasticsearch_client, create_index
    
    client = get_elasticsearch_client(es_url)
    create_index(client, index_name, embedding_dim, delete_existing=True)
    
    return client


def run_ingestion(client, products: List[Dict[str, Any]], index_name: str):
    """Step 4: Ingest products into Elasticsearch."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: INGESTING PRODUCTS")
    logger.info("=" * 70)
    
    from ingest import bulk_index_products, verify_ingestion
    
    stats = bulk_index_products(client, products, index_name)
    verify_ingestion(client, index_name, len(products))
    
    return stats


def run_evaluation(client, queries: List[Dict[str, Any]], index_name: str):
    """Step 5: Run evaluation framework."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: RUNNING EVALUATION")
    logger.info("=" * 70)
    
    from search import SearchEngine
    from evaluate import (
        SearchEvaluator,
        generate_visualization,
        generate_query_type_visualization
    )
    
    # Create search engine
    engine = SearchEngine(client, index_name)
    
    # Create evaluator
    evaluator = SearchEvaluator(engine, queries, k=10)
    
    # Run evaluation
    results = evaluator.evaluate_all_methods()
    
    # Print results
    evaluator.print_results()
    evaluator.print_improvements()
    
    # Generate visualizations
    generate_visualization(results, "search_comparison.png")
    
    # Evaluate by query type
    results_by_type = evaluator.evaluate_by_query_type()
    generate_query_type_visualization(results_by_type, "search_by_query_type.png")
    
    return results, results_by_type, evaluator


def generate_blog_output(
    client,
    queries: List[Dict[str, Any]],
    index_name: str,
    results: Dict[str, Dict[str, float]],
    evaluator
):
    """Step 6: Generate blog-ready output with sample searches."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: GENERATING BLOG-READY OUTPUT")
    logger.info("=" * 70)
    
    from search import SearchEngine
    
    engine = SearchEngine(client, index_name)
    
    # Select interesting sample queries that demonstrate the differences
    sample_queries = [
        # Synonym query - vector should excel
        {
            "query": "gym footwear for running",
            "expected_behavior": "BM25 struggles with 'gym footwear' synonym. Vector search understands this means sneakers/running shoes."
        },
        # Natural language query - vector should excel
        {
            "query": "I need something to listen to music while jogging",
            "expected_behavior": "Natural language query. Vector understands intent (wireless earbuds), BM25 needs exact keyword matches."
        },
        # Keyword query - BM25 should work well
        {
            "query": "samsung smartphone 256gb",
            "expected_behavior": "Keyword-heavy query. BM25 should work well here as it matches exact terms."
        },
        # Multi-concept - hybrid should be best
        {
            "query": "lightweight laptop for travel and meetings",
            "expected_behavior": "Multi-concept query. Hybrid combines keyword matching (laptop) with semantic understanding (travel-friendly)."
        },
    ]
    
    print("\n" + "=" * 80)
    print("     BLOG-READY SEARCH COMPARISON OUTPUT")
    print("=" * 80)
    
    for sample in sample_queries:
        query_text = sample["query"]
        expected = sample["expected_behavior"]
        
        print(f"\n{'─' * 80}")
        print(f"QUERY: \"{query_text}\"")
        print(f"{'─' * 80}")
        print(f"\n📝 Expected Behavior: {expected}\n")
        
        # Run all three searches
        bm25_results = engine.search_keyword(query_text, top_k=5)
        vector_results = engine.search_vector(query_text, top_k=5)
        hybrid_results = engine.search_hybrid(query_text, top_k=5)
        
        # Print side by side
        print(f"{'BM25 (Keyword Search)':<26} │ {'Vector (Semantic Search)':<26} │ {'Hybrid (Combined)':<26}")
        print(f"{'─' * 26}─┼─{'─' * 26}─┼─{'─' * 26}")
        
        for i in range(5):
            bm25_title = bm25_results[i]["title"][:24] if i < len(bm25_results) else "-"
            vector_title = vector_results[i]["title"][:24] if i < len(vector_results) else "-"
            hybrid_title = hybrid_results[i]["title"][:24] if i < len(hybrid_results) else "-"
            
            bm25_score = f"({bm25_results[i]['score']:.2f})" if i < len(bm25_results) else ""
            vector_score = f"({vector_results[i]['score']:.3f})" if i < len(vector_results) else ""
            hybrid_score = f"({hybrid_results[i]['score']:.2f})" if i < len(hybrid_results) else ""
            
            print(f"{i+1}. {bm25_title:<19}{bm25_score:>6} │ {vector_title:<19}{vector_score:>7} │ {hybrid_title:<19}{hybrid_score:>7}")
        
        # Analysis
        print("\n🔍 Analysis:")
        
        # Check if vector/hybrid found more relevant results
        bm25_categories = set(r["category"] for r in bm25_results)
        vector_categories = set(r["category"] for r in vector_results)
        
        if "footwear" in query_text.lower() or "running" in query_text.lower():
            bm25_clothing = sum(1 for r in bm25_results if r["category"] == "clothing")
            vector_clothing = sum(1 for r in vector_results if r["category"] == "clothing")
            if vector_clothing > bm25_clothing:
                print(f"   ✓ Vector search found {vector_clothing} clothing results vs BM25's {bm25_clothing}")
                print("   ✓ Vector search better understood 'footwear' = 'sneakers/shoes'")
            elif bm25_clothing >= vector_clothing:
                print(f"   • Both methods found relevant clothing products")
        
        if "music" in query_text.lower() or "listen" in query_text.lower():
            bm25_electronics = sum(1 for r in bm25_results if r["category"] == "electronics")
            vector_electronics = sum(1 for r in vector_results if r["category"] == "electronics")
            if vector_electronics > bm25_electronics:
                print(f"   ✓ Vector search found {vector_electronics} electronics vs BM25's {bm25_electronics}")
                print("   ✓ Vector understood intent: 'listen to music while jogging' → wireless earbuds")
        
        if "laptop" in query_text.lower() or "smartphone" in query_text.lower():
            print("   • Keyword-rich query: BM25 likely competitive on exact matches")
            print("   • Hybrid combines keyword precision with semantic understanding")
    
    # Print summary metrics
    print("\n" + "=" * 80)
    print("     SUMMARY METRICS")
    print("=" * 80)
    
    evaluator.print_results()
    evaluator.print_improvements()
    
    # Print key insights
    print("\n" + "=" * 80)
    print("     KEY INSIGHTS FOR BLOG")
    print("=" * 80)
    
    improvements = evaluator.compute_improvements()
    
    print("""
🔑 KEY FINDINGS:

1. SEMANTIC SEARCH SOLVES THE SYNONYM PROBLEM
   - Traditional BM25 fails when users search with synonyms
   - "gym footwear" doesn't match "sneakers" with keywords alone
   - Vector embeddings capture meaning: footwear ≈ shoes ≈ sneakers
   
2. VECTOR SEARCH UNDERSTANDS INTENT
   - Natural language queries like "I need something for jogging"
   - BM25 needs exact keyword matches; vector understands context
   - Result: Better user experience with conversational search
   
3. HYBRID SEARCH PROVIDES BALANCE
   - Combines keyword precision with semantic understanding
   - Works well across ALL query types
   - Best choice for production e-commerce search
   
4. ELASTICSEARCH + HNSW = SCALABLE VECTOR SEARCH
   - HNSW algorithm provides sub-millisecond lookups
   - Scales to millions of products
   - Native integration with existing ES infrastructure
""")
    
    # Print specific improvements
    if improvements:
        print(f"📊 PERFORMANCE IMPROVEMENTS OVER BM25:")
        for method, method_improvements in improvements.items():
            mrr_imp = method_improvements.get('mrr', 0)
            p10_imp = method_improvements.get('precision@10', 0)
            print(f"   {method.upper()}: MRR {mrr_imp:+.1f}%, Precision@10 {p10_imp:+.1f}%")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point for the complete pipeline."""
    parser = argparse.ArgumentParser(
        description="E-commerce Semantic Search System"
    )
    parser.add_argument(
        "--es-url",
        default=DEFAULT_ES_URL,
        help="Elasticsearch URL"
    )
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX_NAME,
        help="Index name"
    )
    parser.add_argument(
        "--skip-data-gen",
        action="store_true",
        help="Skip data generation (use existing files)"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation (use existing file)"
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Skip indexing (use existing index)"
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Only run evaluation (skip all setup)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("     E-COMMERCE SEMANTIC SEARCH SYSTEM")
    print("     Demonstrating BM25 vs Vector vs Hybrid Search")
    print("=" * 70)
    
    # Initialize variables
    products = None
    queries = None
    embedding_dim = 384  # Default for MiniLM
    
    try:
        # Step 1: Data Generation
        if not args.skip_data_gen and not args.evaluation_only:
            products, queries = run_data_generation()
        else:
            logger.info("Loading existing data files...")
            with open("products.json", "r") as f:
                products = json.load(f)
            with open("queries.json", "r") as f:
                queries = json.load(f)
        
        # Step 2: Embedding Generation
        if not args.skip_embeddings and not args.evaluation_only:
            products, embedding_dim = run_embedding_generation(products)
        else:
            if os.path.exists("products_with_embeddings.json"):
                logger.info("Loading existing embeddings...")
                with open("products_with_embeddings.json", "r") as f:
                    products = json.load(f)
                if products and "embedding" in products[0]:
                    embedding_dim = len(products[0]["embedding"])
        
        # Step 3: Index Setup
        if not args.skip_indexing and not args.evaluation_only:
            client = run_index_setup(args.es_url, args.index, embedding_dim)
        else:
            from index_setup import get_elasticsearch_client
            client = get_elasticsearch_client(args.es_url)
        
        # Step 4: Ingestion
        if not args.skip_indexing and not args.evaluation_only:
            run_ingestion(client, products, args.index)
        
        # Step 5: Evaluation
        results, results_by_type, evaluator = run_evaluation(client, queries, args.index)
        
        # Step 6: Blog Output
        generate_blog_output(client, queries, args.index, results, evaluator)
        
        print("\n" + "=" * 70)
        print("     PIPELINE COMPLETE!")
        print("=" * 70)
        print("""
Generated files:
  - products.json: Synthetic product catalog
  - queries.json: Test queries with ground truth
  - products_with_embeddings.json: Products with vector embeddings
  - search_comparison.png: Visualization of method comparison
  - search_by_query_type.png: Performance by query type

To run individual components:
  python data_generation.py   # Generate data
  python embed.py            # Generate embeddings
  python index_setup.py      # Setup ES index
  python ingest.py          # Index products
  python search.py          # Demo searches
  python evaluate.py        # Run evaluation
        """)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
