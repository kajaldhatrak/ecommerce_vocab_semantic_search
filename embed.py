"""
Embedding Generation Module for E-commerce Semantic Search System
==================================================================

This module generates dense vector embeddings using sentence-transformers.

Why We Use Dense Vector Embeddings:
-----------------------------------
Traditional keyword search (BM25) matches exact words and their frequencies.
It fails when users search with synonyms or paraphrases:
- "sneakers" won't match "running shoes" 
- "mobile phone" won't match "smartphone"
- "gym footwear" won't match "athletic trainers"

Dense vector embeddings solve this by:
1. Converting text into high-dimensional vectors (384-768 dimensions)
2. Placing semantically similar texts close together in vector space
3. Enabling similarity search based on MEANING, not just keywords

The all-MiniLM-L6-v2 model is ideal because:
- Small and fast (22M parameters)
- Good quality embeddings (384 dimensions)
- Trained on 1B+ sentence pairs
- Maps sentences & paragraphs to dense vectors
"""

import json
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configuration
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingGenerator:
    """
    Generates dense vector embeddings using sentence-transformers.
    
    This class encapsulates the embedding model and provides efficient
    batch processing for large datasets.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformer model to use.
                       Default is 'all-MiniLM-L6-v2' which provides a good
                       balance of speed and quality.
        """
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
        # Auto-detect embedding dimension from model
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension of the loaded model."""
        return self.embedding_dim
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
        
        Returns:
            List of floats representing the embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Uses batch processing to maximize GPU/CPU utilization.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar
        
        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings.tolist()
    
    def combine_text_fields(self, title: str, description: str) -> str:
        """
        Combine title and description for embedding generation.
        
        We combine these fields because:
        - Title provides key product identifiers
        - Description provides semantic context
        - Together they give a complete representation
        
        Args:
            title: Product title
            description: Product description
        
        Returns:
            Combined text for embedding
        """
        return f"{title}. {description}"


def generate_product_embeddings(
    products: List[Dict[str, Any]],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32
) -> tuple[List[Dict[str, Any]], int]:
    """
    Generate embeddings for all products.
    
    Combines title + description for each product and generates
    dense vector embeddings. This is the key preprocessing step
    that enables semantic search.
    
    Args:
        products: List of product dictionaries
        model_name: Sentence transformer model to use
        batch_size: Batch size for efficient processing
    
    Returns:
        Tuple of (products with embeddings, embedding dimension)
    """
    logger.info("=" * 60)
    logger.info("Generating Product Embeddings")
    logger.info("=" * 60)
    
    # Initialize embedding generator
    generator = EmbeddingGenerator(model_name)
    embedding_dim = generator.get_embedding_dimension()
    
    # Prepare texts for embedding (combine title + description)
    logger.info("Preparing product texts...")
    texts = []
    for product in products:
        combined_text = generator.combine_text_fields(
            product["title"],
            product["description"]
        )
        texts.append(combined_text)
    
    # Generate embeddings in batches
    logger.info(f"Generating embeddings with batch size {batch_size}...")
    embeddings = generator.generate_embeddings_batch(texts, batch_size=batch_size)
    
    # Attach embeddings to products
    logger.info("Attaching embeddings to products...")
    for product, embedding in zip(products, embeddings):
        product["embedding"] = embedding
    
    logger.info(f"Generated {len(embeddings)} embeddings of dimension {embedding_dim}")
    
    return products, embedding_dim


def generate_query_embedding(
    query: str,
    generator: Optional[EmbeddingGenerator] = None,
    model_name: str = DEFAULT_MODEL_NAME
) -> List[float]:
    """
    Generate embedding for a search query.
    
    This uses the same model as product embeddings to ensure
    queries and products are in the same vector space.
    
    Args:
        query: Search query text
        generator: Pre-initialized EmbeddingGenerator (optional)
        model_name: Model name if generator not provided
    
    Returns:
        Query embedding vector
    """
    if generator is None:
        generator = EmbeddingGenerator(model_name)
    
    return generator.generate_embedding(query)


def load_products(filepath: str = "products.json") -> List[Dict[str, Any]]:
    """Load products from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_products_with_embeddings(
    products: List[Dict[str, Any]],
    filepath: str = "products_with_embeddings.json"
) -> str:
    """
    Save products with embeddings to JSON file.
    
    Note: Embeddings are large, so this file will be significantly
    bigger than the original products.json
    
    Args:
        products: Products with embedding vectors
        filepath: Output file path
    
    Returns:
        Path to saved file
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved products with embeddings to {filepath}")
    return filepath


def main():
    """Main function to generate and save embeddings."""
    logger.info("=" * 60)
    logger.info("E-commerce Semantic Search - Embedding Generation")
    logger.info("=" * 60)
    
    # Load products
    products = load_products("products.json")
    logger.info(f"Loaded {len(products)} products")
    
    # Generate embeddings
    products_with_embeddings, embedding_dim = generate_product_embeddings(products)
    
    # Save to file
    save_products_with_embeddings(products_with_embeddings)
    
    # Print statistics
    logger.info("\n" + "=" * 60)
    logger.info("Embedding Statistics:")
    logger.info("=" * 60)
    logger.info(f"Total products: {len(products_with_embeddings)}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Total embedding values: {len(products_with_embeddings) * embedding_dim:,}")
    
    # Show sample embedding
    sample_product = products_with_embeddings[0]
    logger.info(f"\nSample Product: {sample_product['title']}")
    logger.info(f"Embedding (first 10 values): {sample_product['embedding'][:10]}")
    
    logger.info("\nEmbedding generation complete!")


if __name__ == "__main__":
    main()
