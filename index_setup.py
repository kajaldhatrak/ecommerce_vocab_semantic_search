"""
Elasticsearch Index Setup Module
================================

This module handles creation and configuration of the Elasticsearch index
for the e-commerce semantic search system.

Index Design Rationale:
-----------------------

1. TEXT FIELDS (for BM25 keyword search):
   - title: Analyzed for full-text search
   - description: Analyzed for full-text search
   BM25 scoring considers term frequency, document frequency, and field length.
   This is the traditional search approach that works well for exact matches.

2. KEYWORD FIELDS (for filtering and aggregations):
   - category: Exact match filtering
   - brand: Exact match filtering
   These are NOT analyzed, enabling precise filtering.

3. DENSE_VECTOR FIELD (for semantic search):
   - embedding: Dense vector representation of title + description
   
   Configuration explained:
   - dims: Auto-detected from model (384 for MiniLM)
   - index: true - Enables kNN search using HNSW algorithm
   - similarity: cosine - Best for normalized embeddings from sentence-transformers
   
   HNSW (Hierarchical Navigable Small World):
   - Approximate nearest neighbor algorithm
   - Extremely fast lookups (sub-millisecond for millions of vectors)
   - Small memory footprint with graph-based navigation
   - Trade-off: Slight accuracy loss for massive speed gains

Why COSINE similarity?
- Sentence-transformer embeddings are unit normalized
- Cosine similarity measures direction, not magnitude
- Two texts with similar meaning point in similar directions
- Range: -1 to 1 (we typically see 0.3 to 1.0 for related content)

Hybrid Search Strategy:
-----------------------
By having both text fields (BM25) and dense_vector (semantic), we can:
1. Run BM25 for exact keyword matching
2. Run kNN for semantic similarity
3. Combine scores for best of both worlds
"""

import logging
from typing import Dict, Any, Optional
from elasticsearch import Elasticsearch, exceptions as es_exceptions

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_ES_URL = "http://localhost:9200"
DEFAULT_INDEX_NAME = "ecommerce_products"


def get_elasticsearch_client(
    es_url: str = DEFAULT_ES_URL,
    verify_certs: bool = False,
    timeout: int = 120
) -> Elasticsearch:
    """
    Create and return Elasticsearch client.
    
    Args:
        es_url: Elasticsearch server URL
        verify_certs: Whether to verify SSL certificates
        timeout: Request timeout in seconds (default 120 for bulk operations)
    
    Returns:
        Elasticsearch client instance
    """
    logger.info(f"Connecting to Elasticsearch at {es_url}")
    
    client = Elasticsearch(
        es_url,
        verify_certs=verify_certs,
        request_timeout=timeout,
        retry_on_timeout=True,
        max_retries=3
    )
    
    # Verify connection
    if client.ping():
        info = client.info()
        logger.info(f"Connected to Elasticsearch cluster: {info['cluster_name']}")
        logger.info(f"Elasticsearch version: {info['version']['number']}")
        return client
    else:
        raise ConnectionError(f"Could not connect to Elasticsearch at {es_url}")


def create_index_mapping(embedding_dim: int) -> Dict[str, Any]:
    """
    Create the index mapping with text, keyword, and dense_vector fields.
    
    This mapping is optimized for hybrid search combining:
    - BM25 text search on title and description
    - kNN vector search on embeddings
    - Filtered queries on category, brand, price
    
    Args:
        embedding_dim: Dimension of embedding vectors (auto-detected from model)
    
    Returns:
        Complete index mapping configuration
    """
    mapping = {
        "settings": {
            # Optimize for search performance
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "1s",
            # Analysis settings for better text matching
            "analysis": {
                "analyzer": {
                    "product_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "stop", "snowball"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                # ============ Core Product Fields ============
                "id": {
                    "type": "keyword"
                },
                
                # TEXT fields for BM25 search
                # These are analyzed (tokenized, lowercased, stemmed)
                "title": {
                    "type": "text",
                    "analyzer": "product_analyzer",
                    # Store a keyword version for sorting and aggregations
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "description": {
                    "type": "text",
                    "analyzer": "product_analyzer"
                },
                
                # KEYWORD fields for exact matching and filtering
                # These are NOT analyzed - exact matches only
                "category": {
                    "type": "keyword"
                },
                "brand": {
                    "type": "keyword"
                },
                "product_type": {
                    "type": "keyword"
                },
                
                # Numeric fields for range queries
                "price": {
                    "type": "float"
                },
                "rating": {
                    "type": "float"
                },
                
                # Nested object for flexible attributes
                "attributes": {
                    "type": "object",
                    "enabled": True
                },
                
                # ============ Dense Vector Field ============
                # This is the key field for semantic search
                "embedding": {
                    "type": "dense_vector",
                    # Dynamic dimension based on model
                    "dims": embedding_dim,
                    # CRITICAL: index must be true for kNN search
                    "index": True,
                    # Cosine similarity for normalized embeddings
                    "similarity": "cosine",
                    # HNSW configuration for approximate nearest neighbor
                    # m: Number of connections per node (higher = more accurate, more memory)
                    # ef_construction: Build-time beam width (higher = better quality)
                    "index_options": {
                        "type": "hnsw",
                        "m": 16,
                        "ef_construction": 100
                    }
                }
            }
        }
    }
    
    return mapping


def delete_index_if_exists(
    client: Elasticsearch,
    index_name: str = DEFAULT_INDEX_NAME
) -> bool:
    """
    Delete index if it exists (for clean recreation).
    
    Args:
        client: Elasticsearch client
        index_name: Name of the index to delete
    
    Returns:
        True if index was deleted, False if it didn't exist
    """
    try:
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)
            logger.info(f"Deleted existing index: {index_name}")
            return True
        else:
            logger.info(f"Index {index_name} does not exist, nothing to delete")
            return False
    except es_exceptions.ApiError as e:
        logger.error(f"Error deleting index: {e}")
        raise


def create_index(
    client: Elasticsearch,
    index_name: str = DEFAULT_INDEX_NAME,
    embedding_dim: int = 384,
    delete_existing: bool = True
) -> bool:
    """
    Create the e-commerce product index with proper mappings.
    
    This sets up the index structure for hybrid search:
    - Text fields analyzed for BM25
    - Dense vector field indexed for kNN
    
    Args:
        client: Elasticsearch client
        index_name: Name of the index to create
        embedding_dim: Dimension of embedding vectors
        delete_existing: Whether to delete existing index first
    
    Returns:
        True if index was created successfully
    """
    logger.info("=" * 60)
    logger.info(f"Creating Elasticsearch Index: {index_name}")
    logger.info("=" * 60)
    
    # Delete existing index if requested
    if delete_existing:
        delete_index_if_exists(client, index_name)
    
    # Create index with mapping
    mapping = create_index_mapping(embedding_dim)
    
    try:
        client.indices.create(
            index=index_name,
            body=mapping
        )
        logger.info(f"Successfully created index: {index_name}")
        logger.info(f"Embedding dimension: {embedding_dim}")
        
        # Log mapping details
        logger.info("\nIndex configuration:")
        logger.info(f"  - Text fields: title, description (BM25 searchable)")
        logger.info(f"  - Keyword fields: category, brand, id (exact match)")
        logger.info(f"  - Numeric fields: price, rating (range queries)")
        logger.info(f"  - Dense vector: embedding ({embedding_dim}D, cosine, HNSW)")
        
        return True
        
    except es_exceptions.ApiError as e:
        logger.error(f"Failed to create index: {e}")
        raise


def verify_index(
    client: Elasticsearch,
    index_name: str = DEFAULT_INDEX_NAME
) -> Dict[str, Any]:
    """
    Verify index exists and return its configuration.
    
    Args:
        client: Elasticsearch client
        index_name: Name of the index to verify
    
    Returns:
        Index mapping and settings
    """
    try:
        # Check if index exists
        if not client.indices.exists(index=index_name):
            raise ValueError(f"Index {index_name} does not exist")
        
        # Get mapping
        mapping = client.indices.get_mapping(index=index_name)
        
        # Get settings
        settings = client.indices.get_settings(index=index_name)
        
        # Get stats
        stats = client.indices.stats(index=index_name)
        doc_count = stats["indices"][index_name]["primaries"]["docs"]["count"]
        
        logger.info(f"\nIndex '{index_name}' verification:")
        logger.info(f"  - Document count: {doc_count}")
        logger.info(f"  - Status: Active")
        
        return {
            "mapping": mapping,
            "settings": settings,
            "doc_count": doc_count
        }
        
    except es_exceptions.ApiError as e:
        logger.error(f"Failed to verify index: {e}")
        raise


def setup_index(
    es_url: str = DEFAULT_ES_URL,
    index_name: str = DEFAULT_INDEX_NAME,
    embedding_dim: int = 384
) -> tuple[Elasticsearch, str]:
    """
    Complete index setup workflow.
    
    Args:
        es_url: Elasticsearch URL
        index_name: Index name to create
        embedding_dim: Embedding dimension
    
    Returns:
        Tuple of (Elasticsearch client, index name)
    """
    # Connect to Elasticsearch
    client = get_elasticsearch_client(es_url)
    
    # Create index
    create_index(client, index_name, embedding_dim)
    
    return client, index_name


def main():
    """Main function to setup the Elasticsearch index."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Elasticsearch index")
    parser.add_argument("--es-url", default=DEFAULT_ES_URL, help="Elasticsearch URL")
    parser.add_argument("--index", default=DEFAULT_INDEX_NAME, help="Index name")
    parser.add_argument("--embedding-dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--delete-existing", action="store_true", default=True, help="Delete existing index")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("E-commerce Semantic Search - Index Setup")
    logger.info("=" * 60)
    
    # Connect and create index
    client = get_elasticsearch_client(args.es_url)
    create_index(
        client,
        args.index,
        args.embedding_dim,
        args.delete_existing
    )
    
    # Verify
    verify_index(client, args.index)
    
    logger.info("\nIndex setup complete!")


if __name__ == "__main__":
    main()
