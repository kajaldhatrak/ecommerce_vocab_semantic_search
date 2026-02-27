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