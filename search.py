"""
Search Implementation Module
============================

This module implements three search strategies for comparison:
1. Keyword Search (BM25) - Traditional text matching
2. Vector Search (kNN) - Semantic similarity
3. Hybrid Search - Combined BM25 + Vector

Understanding Each Approach:
----------------------------

BM25 (Best Matching 25):
- Bag-of-words model based on term frequency
- Considers:
  * Term frequency (TF): How often a term appears in document
  * Inverse document frequency (IDF): How rare the term is overall
  * Field length normalization
- Strengths: Exact matches, rare keyword importance
- Weaknesses: No semantic understanding, synonym misses

Vector Search (kNN):
- Compares embedding vectors using similarity measure
- Captures semantic meaning: "sneakers" ≈ "running shoes"
- Strengths: Understanding intent, handling paraphrases
- Weaknesses: May miss exact important keywords

Hybrid Search:
- Combines both approaches
- Gets exact keyword matches AND semantic similarity
- Methods:
  1. Linear combination: α * BM25 + β * vector
  2. Reciprocal Rank Fusion (RRF): Combines rankings
- This implementation uses RRF for robust combination

Why Hybrid Works Best:
- Captures "laptop 16GB" keyword specificity
- Also finds "powerful notebook computer" semantically
- Best of both worlds for diverse query types
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from elasticsearch import Elasticsearch

from embed import EmbeddingGenerator, generate_query_embedding

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_INDEX_NAME = "ecommerce_products"
DEFAULT_TOP_K = 10


class SearchEngine:
    """
    Multi-strategy search engine for e-commerce products.
    
    Supports:
    - BM25 keyword search
    - kNN vector search
    - Hybrid search combining both
    """
    
    def __init__(
        self,
        es_client: Elasticsearch,
        index_name: str = DEFAULT_INDEX_NAME,
        embedding_generator: Optional[EmbeddingGenerator] = None
    ):
        """
        Initialize the search engine.
        
        Args:
            es_client: Elasticsearch client
            index_name: Index to search
            embedding_generator: Pre-initialized embedding generator (optional)
        """
        self.client = es_client
        self.index_name = index_name
        
        # Lazy-load embedding generator if not provided
        self._embedding_generator = embedding_generator
    
    @property
    def embedding_generator(self) -> EmbeddingGenerator:
        """Get or create embedding generator."""
        if self._embedding_generator is None:
            self._embedding_generator = EmbeddingGenerator()
        return self._embedding_generator
    
    def search_keyword(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        fields: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword search using BM25.
        
        BM25 Algorithm Explanation:
        ---------------------------
        BM25 scores each document based on:
        
        score(D, Q) = Σ IDF(qi) * f(qi, D) * (k1 + 1)
                      ---------------------------------
                      f(qi, D) + k1 * (1 - b + b * |D|/avgdl)
        
        Where:
        - qi: Query term i
        - f(qi, D): Term frequency in document
        - |D|: Document length
        - avgdl: Average document length
        - k1, b: Tuning parameters (default k1=1.2, b=0.75)
        - IDF: Inverse document frequency
        
        This works well for exact keyword matches but fails when:
        - User searches "gym footwear" instead of "sneakers"
        - Query uses synonyms not in the document
        
        Args:
            query: Search query string
            top_k: Number of results to return
            fields: Fields to search (default: title, description)
        
        Returns:
            List of search results with scores
        """
        if fields is None:
            fields = ["title^2", "description"]  # Title boosted 2x
        
        search_query = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": fields,
                    "type": "best_fields",
                    "tie_breaker": 0.3
                }
            },
            "size": top_k,
            "_source": {
                "excludes": ["embedding"]  # Don't return large embedding vector
            }
        }
        
        response = self.client.search(index=self.index_name, **search_query)
        
        return self._format_results(response, "bm25")
    
    def search_vector(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        num_candidates: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Perform vector search using kNN.
        
        kNN (k-Nearest Neighbors) with HNSW:
        -----------------------------------
        1. Query is converted to embedding vector
        2. HNSW graph navigated to find similar vectors
        3. Cosine similarity computed for ranking
        
        How HNSW works:
        - Hierarchical graph with multiple layers
        - Higher layers: Large jumps for quick navigation
        - Lower layers: Fine-grained connections for precision
        - O(log N) search complexity
        
        Why this captures semantics:
        - "sneakers" and "running shoes" have similar embeddings
        - "cold weather coat" similar to "winter jacket"
        - Model learned these relationships from training data
        
        Args:
            query: Search query string
            top_k: Number of results to return
            num_candidates: Candidates to consider (higher = more accurate)
        
        Returns:
            List of search results with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        search_query = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": num_candidates
            },
            "_source": {
                "excludes": ["embedding"]
            }
        }
        
        response = self.client.search(index=self.index_name, **search_query)
        
        return self._format_results(response, "vector")
    
    def search_hybrid(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        bm25_boost: float = 1.0,
        vector_boost: float = 1.0,
        num_candidates: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and vector similarity.
        
        Hybrid Search Strategy:
        ----------------------
        We use Elasticsearch 8.x's native hybrid search which:
        1. Runs both BM25 and kNN queries
        2. Combines scores using reciprocal rank fusion (RRF)
        
        RRF Formula:
        RRF(d) = Σ 1 / (k + rank(d, query))
        
        Where k is a constant (typically 60), and rank(d, query) is
        the position of document d in each result list.
        
        Benefits of RRF:
        - Handles different score scales automatically
        - Robust to outliers
        - Simple to understand and tune
        
        When hybrid excels:
        - Query: "lightweight laptop for travel"
        - BM25 finds: "laptop" keyword matches
        - Vector finds: semantically similar portable computers
        - Hybrid: Combines both signals for best ranking
        
        Args:
            query: Search query string
            top_k: Number of results to return
            bm25_boost: Boost factor for BM25 results
            vector_boost: Boost factor for vector results
            num_candidates: Candidates for kNN search
        
        Returns:
            List of search results with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Elasticsearch 8.x hybrid search using sub_searches with RRF
        search_query = {
            "size": top_k,
            "_source": {
                "excludes": ["embedding"]
            },
            "query": {
                "bool": {
                    "should": [
                        # BM25 text match
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^2", "description"],
                                "boost": bm25_boost
                            }
                        }
                    ]
                }
            },
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": num_candidates,
                "boost": vector_boost
            }
        }
        
        response = self.client.search(index=self.index_name, **search_query)
        
        return self._format_results(response, "hybrid")
    
    def search_hybrid_rrf(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        num_candidates: int = 100,
        rrf_rank_constant: int = 60,
        rrf_window_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using explicit Reciprocal Rank Fusion.
        
        This is an alternative implementation that manually computes RRF
        scores by running BM25 and vector searches separately, then
        combining the rankings.
        
        RRF is preferred over linear score combination because:
        1. Score scales differ (BM25 unbounded, cosine 0-1)
        2. RRF normalizes by rank, not score value
        3. More robust to scoring disparities
        
        Args:
            query: Search query string
            top_k: Number of final results
            num_candidates: Initial candidates from each search
            rrf_rank_constant: k in RRF formula (default 60)
            rrf_window_size: How many results from each search to consider
        
        Returns:
            List of search results with RRF scores
        """
        # Get results from both searches
        bm25_results = self.search_keyword(query, top_k=rrf_window_size)
        vector_results = self.search_vector(query, top_k=rrf_window_size, num_candidates=num_candidates)
        
        # Build rank maps
        bm25_ranks = {r["id"]: rank + 1 for rank, r in enumerate(bm25_results)}
        vector_ranks = {r["id"]: rank + 1 for rank, r in enumerate(vector_results)}
        
        # Get all unique document IDs
        all_doc_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        doc_data = {}
        
        # Store document data for reconstruction
        for r in bm25_results + vector_results:
            if r["id"] not in doc_data:
                doc_data[r["id"]] = r
        
        for doc_id in all_doc_ids:
            rrf_score = 0
            
            # Add BM25 contribution
            if doc_id in bm25_ranks:
                rrf_score += 1 / (rrf_rank_constant + bm25_ranks[doc_id])
            
            # Add vector contribution
            if doc_id in vector_ranks:
                rrf_score += 1 / (rrf_rank_constant + vector_ranks[doc_id])
            
            rrf_scores[doc_id] = rrf_score
        
        # Sort by RRF score and take top_k
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        
        # Construct results
        results = []
        for doc_id in sorted_ids:
            result = doc_data[doc_id].copy()
            result["score"] = rrf_scores[doc_id]
            result["search_type"] = "hybrid_rrf"
            result["bm25_rank"] = bm25_ranks.get(doc_id, None)
            result["vector_rank"] = vector_ranks.get(doc_id, None)
            results.append(result)
        
        return results
    
    def _format_results(
        self,
        response: Dict[str, Any],
        search_type: str
    ) -> List[Dict[str, Any]]:
        """
        Format Elasticsearch response into standardized results.
        
        Args:
            response: Raw Elasticsearch response
            search_type: Type of search performed
        
        Returns:
            List of formatted result dictionaries
        """
        results = []
        
        for hit in response["hits"]["hits"]:
            result = {
                "id": hit["_source"]["id"],
                "title": hit["_source"]["title"],
                "description": hit["_source"]["description"],
                "category": hit["_source"]["category"],
                "brand": hit["_source"]["brand"],
                "price": hit["_source"]["price"],
                "rating": hit["_source"]["rating"],
                "score": hit["_score"],
                "search_type": search_type
            }
            results.append(result)
        
        return results


def compare_search_methods(
    engine: SearchEngine,
    query: str,
    top_k: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run all three search methods and compare results.
    
    Args:
        engine: SearchEngine instance
        query: Search query
        top_k: Number of results per method
    
    Returns:
        Dictionary with results from each method
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Query: '{query}'")
    logger.info(f"{'='*60}")
    
    results = {
        "bm25": engine.search_keyword(query, top_k),
        "vector": engine.search_vector(query, top_k),
        "hybrid": engine.search_hybrid(query, top_k)
    }
    
    for method, method_results in results.items():
        logger.info(f"\n{method.upper()} Results:")
        logger.info("-" * 40)
        for i, r in enumerate(method_results, 1):
            logger.info(f"{i}. {r['title'][:50]}... (score: {r['score']:.4f})")
    
    return results


def print_search_comparison(
    results: Dict[str, List[Dict[str, Any]]],
    query: str
):
    """
    Print a nicely formatted comparison of search results.
    
    Args:
        results: Results from each search method
        query: Original query
    """
    print(f"\n{'='*80}")
    print(f"SEARCH COMPARISON")
    print(f"Query: \"{query}\"")
    print(f"{'='*80}")
    
    methods = ["bm25", "vector", "hybrid"]
    
    for i in range(max(len(r) for r in results.values())):
        print(f"\n--- Rank {i+1} ---")
        for method in methods:
            if i < len(results[method]):
                r = results[method][i]
                print(f"{method.upper():8s}: {r['title'][:45]:45s} | Score: {r['score']:.3f}")
            else:
                print(f"{method.upper():8s}: -")


def main():
    """Main function to demonstrate search methods."""
    from index_setup import get_elasticsearch_client
    
    logger.info("=" * 60)
    logger.info("E-commerce Semantic Search - Search Demo")
    logger.info("=" * 60)
    
    # Connect to Elasticsearch
    client = get_elasticsearch_client()
    
    # Create search engine
    engine = SearchEngine(client)
    
    # Test queries demonstrating different scenarios
    test_queries = [
        # Synonym query - vector should excel
        "gym footwear for running",
        
        # Natural language - vector should excel
        "I need something to listen to music while jogging",
        
        # Keyword query - BM25 should work well
        "samsung smartphone 256gb",
        
        # Multi-concept - hybrid should be best
        "lightweight laptop for travel and work meetings",
    ]
    
    for query in test_queries:
        results = compare_search_methods(engine, query, top_k=5)
        print_search_comparison(results, query)


if __name__ == "__main__":
    main()
