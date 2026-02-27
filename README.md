**Project Overview**
- **What:** A compact e‑commerce semantic search demo (BM25 + vector + hybrid) using Python and Elasticsearch 8.x.
- **Purpose:** Reproducible demo and learning playground showing synthetic product generation, sentence-transformer embeddings, Elasticsearch dense_vector (HNSW) indexing, and evaluation (Precision@10, MRR).

**Prerequisites**
- **Python:** 3.9+ (virtualenv recommended).
- **Elasticsearch:** Running Elasticsearch 8.x at `http://localhost:9200` (single-node dev cluster is fine). Ensure the cluster has sufficient heap (e.g., 4GB) for indexing vectors.
- **Network:** Internet required for downloading the embedding model on first run (SentenceTransformers).

**Quick Setup (Windows PowerShell)**
1. Create & activate venv:

```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start Elasticsearch (local) and confirm it's reachable:

```powershell
curl http://localhost:9200
```

**How to run the full pipeline**
- This repository contains small scripts to run each stage. From the repository root with the venv active:

```powershell
# generate synthetic products and queries (only if you want to regenerate data)
python data_generation.py

# embed products using sentence-transformers (saves products_with_embeddings.json)
python embed.py

# create index with dense_vector mapping
python index_setup.py

# ingest documents into Elasticsearch
python ingest.py

# run example searches (BM25 / vector / hybrid)
python search.py

# evaluate retrieval performance and produce visuals
python evaluate.py

# or run orchestrator to run stages with flags
python main.py --help
```

**Venv creation notes (POSIX shells)**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Technical Details (low-level)**
- **Data generation:** `data_generation.py` creates a synthetic product catalog (title, description, category, specs) and a small set of queries. Product text fields are concatenated into a single `combined_text` used for embedding.
- **Embeddings:** `embed.py` uses `sentence-transformers` (default model: `all-MiniLM-L6-v2`) to encode `combined_text` into 384-dimensional dense vectors. Embeddings are saved alongside product JSON in `products_with_embeddings.json`.
- **Index mapping:** `index_setup.py` creates an index `ecommerce_products` with a `dense_vector` field (dims=384) and HNSW params (e.g., `m=16`, `ef_construction=100`) and a standard text mapping for BM25. Similarity is configured for cosine when using vectors.
- **Ingest:** `ingest.py` performs bulk indexing. For reliability the demo reduces batch size and sets request timeouts. If your ES times out, increase `request_timeout` or reduce `BATCH_SIZE`.
- **Search modes:** `search.py` implements three methods:
  - BM25: standard text multi_match queries across title/description.
  - Vector: kNN search against the `dense_vector` field (HNSW / kNN API).
  - Hybrid: fused score combining BM25 and vector scores.
- **Evaluation:** `evaluate.py` runs the queries against each method and computes Precision@10 and MRR (and other metrics). Visuals are saved as PNGs.

**Configuration**
- ES endpoint and client timeouts are defined in `index_setup.py` and `ingest.py` (adjust `ELASTIC_URL`, `request_timeout`, and `max_retries` if needed).
- Embedding model is configured in `embed.py` (change `MODEL_NAME` to try other SentenceTransformers models).

**Files you should push to GitHub**
- Core code & docs (recommended):
  - [requirements.txt](requirements.txt)
  - [data_generation.py](data_generation.py)
  - [embed.py](embed.py)
  - [index_setup.py](index_setup.py)
  - [ingest.py](ingest.py)
  - [search.py](search.py)
  - [evaluate.py](evaluate.py)
  - [main.py](main.py)
  - [BLOG_ARTICLE.md](BLOG_ARTICLE.md)
  - [README.md](README.md) (this file)

- Optional add-ins (small / useful):
  - [products.json](products.json) (synthetic catalog — if small)
  - [queries.json](queries.json) (test queries)

- Files you should NOT push
  - The virtual environment folder `.venv/` (add to `.gitignore`).
  - Large binary data / embeddings (e.g., `products_with_embeddings.json`) — instead regenerate embeddings using `embed.py` on first run.

**Troubleshooting tips**
- If embedding download fails: ensure internet access and enough disk space; large models may take time.
- If ES bulk index times out: increase `request_timeout` in `ingest.py`, reduce `BATCH_SIZE`, or increase Elasticsearch heap size.
- If vector search returns poor results: try a different embedding model (`all-mpnet-base-v2`), tune hybrid weight, or inspect queries for ambiguity.

**Commit & push example (safe minimal push)**

```powershell
git init
git add requirements.txt data_generation.py embed.py index_setup.py ingest.py search.py evaluate.py main.py BLOG_ARTICLE.md README.md
git commit -m "Add semantic e-commerce demo code and docs"
git remote add origin <your-repo-url>
git push -u origin main
```

**Next steps I can do for you**
- Add a `.gitignore` with `.venv/` and typical Python ignores.
- Add a small sample `products_sample.json` (trimmed) so GitHub has example data without large files.
- Integrate a query router that routes between BM25/vector/hybrid automatically.

If you want any of the next steps, tell me which and I'll add them.
# E-commerce Semantic Search System

A complete end-to-end demonstration of semantic search using Elasticsearch 8.x with dense vectors (HNSW), comparing BM25 keyword search, vector search, and hybrid search.

## 🎯 Project Goal

Demonstrate that **vector search significantly improves semantic relevance** compared to traditional keyword search:

- **BM25 (Keyword Search)**: Fails on synonyms - "gym footwear" won't match "sneakers"
- **Vector Search (kNN)**: Captures semantic meaning - understands "footwear" ≈ "shoes"
- **Hybrid Search**: Best of both worlds - keyword precision + semantic understanding

## 📁 Project Structure

```
project/
├── data_generation.py      # Generate synthetic e-commerce data
├── embed.py                # Generate embeddings using sentence-transformers
├── index_setup.py          # Create Elasticsearch index with dense_vector
├── ingest.py               # Bulk index products into Elasticsearch
├── search.py               # Implement BM25, vector, and hybrid search
├── evaluate.py             # Evaluation metrics (Precision, Recall, MRR, nDCG)
├── main.py                 # Main orchestrator script
├── requirements.txt        # Python dependencies
├── products.json           # Generated product catalog
├── queries.json            # Test queries with ground truth
└── products_with_embeddings.json  # Products with vector embeddings
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Elasticsearch 8.x running locally on `http://localhost:9200`

### Installation

```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Pipeline

```bash
python main.py
```

This will:
1. Generate 1500 synthetic products
2. Generate 100 test queries with ground truth
3. Create embeddings using `all-MiniLM-L6-v2`
4. Setup Elasticsearch index with dense_vector field
5. Bulk index products
6. Run evaluation comparing all three search methods
7. Generate visualizations and blog-ready output

### Run Individual Components

```bash
python data_generation.py   # Generate synthetic data
python embed.py            # Generate embeddings
python index_setup.py      # Setup Elasticsearch index
python ingest.py          # Index products
python search.py          # Demo searches
python evaluate.py        # Run evaluation
```

## 🔍 Search Methods Explained

### 1. BM25 (Keyword Search)

Traditional bag-of-words search based on term frequency:
- ✅ Works well for exact keyword matches
- ❌ Fails when user searches with synonyms
- ❌ No understanding of semantic meaning

### 2. Vector Search (kNN)

Dense embedding similarity using HNSW algorithm:
- ✅ Captures semantic meaning ("sneakers" ≈ "running shoes")
- ✅ Handles natural language queries
- ❌ May miss important exact keywords

### 3. Hybrid Search

Combines BM25 + Vector using Reciprocal Rank Fusion:
- ✅ Keyword precision + semantic understanding
- ✅ Works well across all query types
- ✅ Best overall performance

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Precision@k** | % of retrieved results that are relevant |
| **Recall@k** | % of relevant items that were retrieved |
| **MRR** | Average reciprocal rank of first relevant result |
| **nDCG@k** | Normalized Discounted Cumulative Gain |

## 🧪 Test Query Types

1. **Synonym Queries**: "gym footwear" instead of "sneakers"
2. **Natural Language**: "I need something to stay warm this winter"
3. **Keyword Queries**: "samsung smartphone 256gb"
4. **Multi-concept**: "lightweight laptop for travel and meetings"

## 📈 Expected Results

| Method | Precision@10 | Recall@10 | MRR |
|--------|--------------|-----------|-----|
| BM25 | Baseline | Baseline | Baseline |
| Vector | +10-30% | +15-40% | +20-50% |
| Hybrid | +15-40% | +20-50% | +25-60% |

*Improvements are typical for datasets with semantic variation*

## 🛠️ Technical Details

### Embedding Model

- **Model**: `all-MiniLM-L6-v2`
- **Dimension**: 384
- **Why**: Small, fast, good quality for semantic similarity

### Elasticsearch Index Configuration

```json
{
  "embedding": {
    "type": "dense_vector",
    "dims": 384,
    "index": true,
    "similarity": "cosine",
    "index_options": {
      "type": "hnsw",
      "m": 16,
      "ef_construction": 100
    }
  }
}
```

### HNSW Parameters

- **m=16**: Connections per node (balance accuracy/memory)
- **ef_construction=100**: Build-time beam width (quality)
- **Similarity=cosine**: Best for normalized embeddings

## 📝 Output Files

After running:
- `search_comparison.png`: Bar chart comparing methods
- `search_by_query_type.png`: Performance by query type

## 🔑 Key Takeaways

1. **Semantic embeddings solve the synonym problem** - "footwear" matches "shoes"
2. **HNSW enables fast similarity search** - Sub-millisecond queries
3. **Hybrid search provides balance** - Best choice for production
4. **Elasticsearch 8.x is vector-ready** - Native dense_vector support

## 📖 Blog Usage

The `main.py` output includes blog-ready content:
- Sample query comparisons
- Side-by-side result tables
- Analysis commentary
- Summary metrics
- Key insights

## License

MIT License - Free for educational and commercial use
