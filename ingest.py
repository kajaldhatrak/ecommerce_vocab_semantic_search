"""
Ingestion Pipeline Module
=========================

This module handles bulk indexing of products into Elasticsearch.

Bulk Indexing Advantages:
-------------------------
Instead of sending individual index requests, we batch multiple documents
together. This provides:
1. Reduced network overhead (fewer HTTP round trips)
2. Better throughput (Elasticsearch optimizes batch writes)
3. Efficient memory usage on the cluster

The typical recommendation is batches of 500-1000 documents.

Error Handling Strategy:
------------------------
- Bulk requests can partially succeed
- We track and log individual document failures
- Failed documents can be retried or logged for investigation
"""

import json
import logging
from typing import List, Dict, Any, Generator, Optional
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_INDEX_NAME = "ecommerce_products"
# Smaller batch size to avoid timeouts with large embedding vectors
DEFAULT_BATCH_SIZE = 100


def generate_bulk_actions(
    products: List[Dict[str, Any]],
    index_name: str = DEFAULT_INDEX_NAME
) -> Generator[Dict[str, Any], None, None]:
    """
    Generate bulk indexing actions for Elasticsearch.
    
    The bulk API expects a specific format:
    - Action metadata (index, document ID)
    - Document source
    
    Using a generator for memory efficiency with large datasets.
    
    Args:
        products: List of products with embeddings
        index_name: Target index name
    
    Yields:
        Bulk action dictionaries
    """
    for product in products:
        # Create the document for indexing
        doc = {
            "id": product["id"],
            "title": product["title"],
            "description": product["description"],
            "category": product["category"],
            "price": product["price"],
            "brand": product["brand"],
            "rating": product["rating"],
            "attributes": product.get("attributes", {}),
            "product_type": product.get("product_type", ""),
            # Dense vector embedding for semantic search
            "embedding": product["embedding"]
        }
        
        yield {
            "_index": index_name,
            "_id": product["id"],
            "_source": doc
        }


def bulk_index_products(
    client: Elasticsearch,
    products: List[Dict[str, Any]],
    index_name: str = DEFAULT_INDEX_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    raise_on_error: bool = False
) -> Dict[str, Any]:
    """
    Bulk index products into Elasticsearch.
    
    This is the main ingestion function that:
    1. Converts products to bulk actions
    2. Sends batched requests to Elasticsearch
    3. Tracks success/failure counts
    4. Logs progress
    
    Args:
        client: Elasticsearch client
        products: List of products with embeddings
        index_name: Target index name
        batch_size: Number of documents per bulk request
        raise_on_error: Whether to raise exceptions on failures
    
    Returns:
        Statistics dictionary with success/failure counts
    """
    logger.info("=" * 60)
    logger.info(f"Bulk Indexing {len(products)} Products")
    logger.info("=" * 60)
    
    stats = {
        "total": len(products),
        "success": 0,
        "failed": 0,
        "errors": []
    }
    
    # Generate bulk actions
    actions = generate_bulk_actions(products, index_name)
    
    # Use streaming_bulk for better memory handling and progress tracking
    logger.info(f"Starting bulk indexing with batch size {batch_size}...")
    
    progress_bar = tqdm(total=len(products), desc="Indexing products")
    
    try:
        # streaming_bulk yields results as they complete
        # request_timeout increased for large embedding payloads
        for success, info in helpers.streaming_bulk(
            client,
            actions,
            chunk_size=batch_size,
            raise_on_error=raise_on_error,
            raise_on_exception=raise_on_error,
            request_timeout=120
        ):
            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1
                stats["errors"].append(info)
                logger.warning(f"Failed to index document: {info}")
            
            progress_bar.update(1)
        
    except Exception as e:
        logger.error(f"Bulk indexing error: {e}")
        raise
    finally:
        progress_bar.close()
    
    # Refresh index to make documents searchable immediately
    logger.info("Refreshing index...")
    client.indices.refresh(index=index_name)
    
    # Log results
    logger.info("\nIngestion Complete!")
    logger.info(f"  ✓ Successfully indexed: {stats['success']}")
    logger.info(f"  ✗ Failed: {stats['failed']}")
    
    if stats["failed"] > 0:
        logger.warning(f"\n{stats['failed']} documents failed to index")
        for error in stats["errors"][:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
    
    return stats


def verify_ingestion(
    client: Elasticsearch,
    index_name: str = DEFAULT_INDEX_NAME,
    expected_count: Optional[int] = None
) -> bool:
    """
    Verify that products were indexed correctly.
    
    Args:
        client: Elasticsearch client
        index_name: Index to verify
        expected_count: Expected document count (optional)
    
    Returns:
        True if verification passes
    """
    logger.info("\nVerifying ingestion...")
    
    # Get document count
    count_response = client.count(index=index_name)
    actual_count = count_response["count"]
    
    logger.info(f"Documents in index: {actual_count}")
    
    if expected_count is not None:
        if actual_count == expected_count:
            logger.info(f"✓ Count matches expected: {expected_count}")
        else:
            logger.warning(f"✗ Count mismatch! Expected {expected_count}, got {actual_count}")
            return False
    
    # Sample a few documents to verify structure
    sample_response = client.search(
        index=index_name,
        query={"match_all": {}},
        size=3
    )
    
    if sample_response["hits"]["total"]["value"] > 0:
        logger.info("\nSample indexed documents:")
        for hit in sample_response["hits"]["hits"]:
            doc = hit["_source"]
            logger.info(f"  - {doc['id']}: {doc['title'][:50]}...")
            
            # Verify embedding exists
            if "embedding" in doc and len(doc["embedding"]) > 0:
                logger.info(f"    Embedding: {len(doc['embedding'])} dimensions ✓")
            else:
                logger.warning(f"    Embedding: MISSING ✗")
                return False
    
    # Test that vector search works
    logger.info("\nTesting vector search capability...")
    try:
        test_query = {
            "knn": {
                "field": "embedding",
                "query_vector": sample_response["hits"]["hits"][0]["_source"]["embedding"],
                "k": 3,
                "num_candidates": 10
            }
        }
        knn_response = client.search(index=index_name, **test_query)
        logger.info(f"✓ kNN search working, returned {len(knn_response['hits']['hits'])} results")
    except Exception as e:
        logger.error(f"✗ kNN search failed: {e}")
        return False
    
    logger.info("\n✓ Ingestion verification complete!")
    return True


def load_products_with_embeddings(filepath: str = "products_with_embeddings.json") -> List[Dict[str, Any]]:
    """
    Load products that have embeddings attached.
    
    Args:
        filepath: Path to products JSON file
    
    Returns:
        List of product dictionaries with embeddings
    """
    logger.info(f"Loading products from {filepath}...")
    
    with open(filepath, "r", encoding="utf-8") as f:
        products = json.load(f)
    
    # Verify embeddings exist
    products_with_embeddings = [p for p in products if "embedding" in p]
    
    if len(products_with_embeddings) != len(products):
        logger.warning(
            f"Only {len(products_with_embeddings)}/{len(products)} products have embeddings"
        )
    
    logger.info(f"Loaded {len(products_with_embeddings)} products with embeddings")
    return products_with_embeddings


def ingest_products(
    client: Elasticsearch,
    products_file: str = "products_with_embeddings.json",
    index_name: str = DEFAULT_INDEX_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> Dict[str, Any]:
    """
    Complete ingestion workflow.
    
    Args:
        client: Elasticsearch client
        products_file: Path to products JSON with embeddings
        index_name: Target index
        batch_size: Bulk batch size
    
    Returns:
        Ingestion statistics
    """
    # Load products
    products = load_products_with_embeddings(products_file)
    
    # Bulk index
    stats = bulk_index_products(client, products, index_name, batch_size)
    
    # Verify
    verify_ingestion(client, index_name, len(products))
    
    return stats


def main():
    """Main function to run the ingestion pipeline."""
    import argparse
    from index_setup import get_elasticsearch_client
    
    parser = argparse.ArgumentParser(description="Ingest products into Elasticsearch")
    parser.add_argument(
        "--products-file",
        default="products_with_embeddings.json",
        help="Path to products JSON file"
    )
    parser.add_argument("--es-url", default="http://localhost:9200", help="Elasticsearch URL")
    parser.add_argument("--index", default=DEFAULT_INDEX_NAME, help="Index name")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("E-commerce Semantic Search - Data Ingestion")
    logger.info("=" * 60)
    
    # Connect to Elasticsearch
    client = get_elasticsearch_client(args.es_url)
    
    # Run ingestion
    stats = ingest_products(
        client,
        args.products_file,
        args.index,
        args.batch_size
    )
    
    logger.info("\nIngestion pipeline complete!")
    logger.info(f"Successfully indexed {stats['success']} products")


if __name__ == "__main__":
    main()
