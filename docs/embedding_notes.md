
# RAG Learning — Embeddings, Similarity Search & FAISS

## 1. Embeddings

Embeddings convert text into numerical vectors representing semantic meaning.

Example:
"JES2 spool usage is high" → [0.12, 0.45, -0.32, ...]

Model: `all-MiniLM-L6-v2`
- Embedding dimensions: 384
- Documents: 10
- Embedding matrix shape: (10, 384)

```python
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    documents,
    normalize_embeddings=True
)
```

### Key Points
- Semantically similar texts tend to have similar vectors.
- Embedding dimensions depend on the model.
- Use compatible query and document embeddings.
- Embeddings represent learned semantic patterns, not factual correctness.

---

## 2. Vector Magnitude

Magnitude represents the length of a vector.

Formula:
||A|| = √(a1² + a2² + ... + an²)

Example:
A = [3, 4]
Magnitude = √(9 + 16) = 5

```python
magnitude = np.linalg.norm(A)
```

---

## 3. Dot Product

Dot product measures vector alignment and is influenced by vector magnitude.

Formula:
A · B = Σ(Ai × Bi)

Example:
A = [3, 4]
B = [6, 8]

Dot Product = (3×6) + (4×8) = 50

```python
dot_product = A @ B
```

- `@` performs matrix multiplication.
- For two 1D vectors, it calculates their dot product.
- Larger magnitudes can produce larger scores.

---

## 4. Cosine Similarity

Cosine similarity measures directional alignment, independent of vector magnitude.

Formula:
cosine(A,B) = (A·B) / (||A|| × ||B||)

Example:
A = [3,4] → magnitude 5
B = [6,8] → magnitude 10

Cosine = 50 / (5×10) = 1

Interpretation:
- 1 → Same direction
- 0 → Perpendicular
- -1 → Opposite direction

Important:
- Higher cosine similarity generally indicates stronger semantic relatedness.
- Similarity does not guarantee document relevance or correctness.

---

## 5. Vector Normalization

Normalization scales a nonzero vector to magnitude 1.

Formula:
A_normalized = A / ||A||

Example:
A = [3,4]
A_normalized = [0.6,0.8]

Magnitude after normalization = 1

```python
normalized = A / np.linalg.norm(A)
```

### Why Normalize?

Cosine Similarity = Dot Product / (Magnitude A × Magnitude B)

After normalization:
- Magnitude A = 1
- Magnitude B = 1

Therefore:

**Normalized Dot Product = Cosine Similarity**

This allows efficient cosine similarity search using inner products.

---

## 6. Semantic Search Without FAISS

Goal: Retrieve the most relevant documents for a query.

```python
query_embedding = model.encode(
    query,
    normalize_embeddings=True
)

scores = embeddings @ query_embedding

top_indices = np.argsort(scores)[::-1][:3]
```

### How It Works

1. Convert documents into embeddings.
2. Convert the query into an embedding.
3. Calculate similarity with every document.
4. Sort scores in descending order.
5. Return Top-K documents.

### Matrix Operations

```python
# Document-to-document similarity
similarity_matrix = embeddings @ embeddings.T

# Query-to-document similarity
scores = embeddings @ query_embedding
```

Shapes:
- Document embeddings: (10,384)
- Query embedding: (384,)
- Similarity output: (10,)

### Limitations
- Every query compares against all documents.
- Computing similarity costs O(N × D).
- Full sorting costs O(N log N).
- NumPy is useful for learning and small datasets.

---

# 7. FAISS — Vector Similarity Search

FAISS = Facebook AI Similarity Search.

FAISS is a library for efficient similarity search over dense vectors.

Developed by Meta AI Research.

### Why FAISS?

Without FAISS:
- Calculate similarities manually.
- Sort documents to find nearest neighbors.

With FAISS:
- Store vectors in a searchable index.
- Search for nearest neighbors using optimized implementations.
- Choose exact or approximate search algorithms.

FAISS is primarily a vector indexing and search library, not a complete vector database.

### FAISS Workflow

Documents
  → Embedding Model
  → Float32 Vectors
  → FAISS Index
  → Query Embedding
  → Nearest-Neighbor Search
  → Top-K Results

---

## 8. FAISS IndexFlatIP

IndexFlatIP performs exact nearest-neighbor search using Inner Product (IP).

```python
import faiss

vectors = embeddings.astype("float32")

dimension = vectors.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(vectors)
```

### Important Properties
- IP = Inner Product.
- Exact search, not approximate.
- No training required.
- Stores full vectors.
- Uses float32 vectors.
- Searches across all indexed vectors.
- Returns highest inner-product scores.

When query and document vectors are normalized:

**Inner Product = Cosine Similarity**

### Search

```python
query_vector = model.encode(
    [query],
    normalize_embeddings=True
).astype("float32")

scores, indices = index.search(query_vector, k=3)
```

Returns:
- `scores`: Similarity values
- `indices`: Matching vector positions

Example:
indices = [[2, 8, 1]]

Corresponding documents:
D3, D9, D2

FAISS returns integer positions, not your original document IDs.

You must maintain mapping between vector positions and documents.

---

## 9. FAISS IndexFlatL2

IndexFlatL2 performs exact nearest-neighbor search using squared Euclidean distance.

Euclidean distance:

distance(A,B) = √Σ(Ai-Bi)²

- Smaller distance = closer vectors.
- FAISS IndexFlatL2 returns squared distances.
- No training required.
- Exact search.

```python
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

distances, indices = index.search(query_vector, k=3)
```

### IP vs L2

| Feature | IndexFlatIP | IndexFlatL2 |
|---|---|---|
| Metric | Inner product | Squared Euclidean |
| Best score | Higher | Lower |
| Exact search | Yes | Yes |
| Training | No | No |
| Cosine search | Normalize vectors | Equivalent ranking if normalized |

For normalized vectors:

Squared L2 Distance = 2 - 2 × Cosine Similarity

Therefore, normalized IP and L2 produce equivalent nearest-neighbor rankings (apart from ties and numerical precision).

---

## 10. Exact Search vs Approximate Search

### Exact Nearest Neighbor (Exact NN)
- Compares against all indexed vectors.
- Finds the true nearest neighbors for the selected metric.
- Higher computational cost for large datasets.

Examples:
- NumPy brute-force similarity
- FAISS IndexFlatIP
- FAISS IndexFlatL2

### Approximate Nearest Neighbor (ANN)
- Avoids exhaustive comparison through specialized index structures.
- Can reduce search latency.
- May miss some true nearest neighbors.
- Introduces recall vs latency trade-offs.

### Common FAISS Indexes

| Index | Approach | Key Trade-off |
|---|---|---|
| Flat | Exhaustive search | Accurate, expensive at scale |
| HNSW | Graph-based ANN | Fast, higher memory usage |
| IVF | Cluster-based ANN | Search fewer clusters |
| IVF-PQ | Clustering + compression | Lower memory, potential quality loss |

### Important Concepts to Learn

HNSW:
- Graph-based nearest-neighbor navigation
- `M`: Graph connectivity
- `efSearch`: Search effort vs recall
- `efConstruction`: Index build effort

IVF:
- Partitions vectors into clusters
- `nlist`: Number of clusters
- `nprobe`: Number of clusters searched
- Requires training

PQ:
- Product Quantization
- Compresses vectors into compact codes
- Reduces memory consumption
- Can reduce retrieval accuracy

---

## 11. NumPy vs FAISS

| Feature | NumPy | FAISS Flat |
|---|---|---|
| Similarity search | Yes | Yes |
| Indexing abstraction | Manual | Built-in |
| Exact nearest neighbor | Yes | Yes |
| Optimized vector search | General-purpose | Specialized |
| ANN support | Not built-in | Yes, with other indexes |
| GPU search | Not native | Supported by FAISS GPU builds |
| Metadata management | Manual | External integration needed |

Important:

FAISS Flat does not automatically reduce search complexity below O(N × D).

Its advantage comes from optimized vector-search implementations and indexing APIs.

For large datasets, use an appropriate ANN index when justified by benchmarks.

---

## 12. Retrieval Evaluation

Retrieval evaluation measures whether the system returns relevant documents.

Notebook example:

Query:
"Why are my batch jobs running slowly?"

Relevant IDs:
D2, D3

Retrieved:
D3, D9, D2

K = 3

### Precision@K

Precision@K = Relevant Retrieved / K

Precision@3 = 2/3 = 0.6667

Meaning:
66.67% of retrieved documents are labeled relevant.

### Recall@K

Recall@K = Relevant Retrieved / Total Relevant

Recall@3 = 2/2 = 1.0

Meaning:
All labeled relevant documents were retrieved.

### Additional Metrics
- MRR: Rank of the first relevant document
- nDCG@K: Ranking quality with graded relevance
- Hit Rate@K: Whether any relevant document appears in Top-K
- ANN Recall@K: Overlap with exact-search ground truth

Important:
- Evaluate with multiple queries.
- Create reliable relevance labels.
- Compare quality, latency, and memory usage.
- Do not conclude overall system quality from one query.

---

## 13. FAISS Limitations

FAISS focuses on vector search.

Additional application components may be needed for:
- Document storage and metadata
- Advanced metadata filtering
- Tenant isolation and access control
- Persistence and index lifecycle
- Distributed operations
- Embedding model management
- Hybrid retrieval and reranking

FAISS does not generate answers or automatically provide a complete RAG pipeline.

---

## 14. RAG Architecture Considerations

### Retrieval Pipeline

Documents
  → Parsing
  → Chunking
  → Embedding
  → Vector Index
  → Query Embedding
  → Retrieval
  → Reranking
  → Context Assembly
  → LLM
  → Grounded Answer

### Important Design Decisions
- Embedding model selection
- Vector dimensions
- Normalization strategy
- Similarity metric: IP, cosine, L2
- Flat vs HNSW vs IVF
- Index memory requirements
- Retrieval latency and throughput
- Exact vs approximate recall
- Metadata filtering and authorization
- Document updates and deletion
- Hybrid search: BM25 + dense retrieval
- Retrieval evaluation and observability

---

## 15. Next Learning Experiments

Completed:
- [x] Generate embeddings
- [x] Normalize vectors
- [x] Calculate dot product and cosine similarity
- [x] Build document similarity matrix
- [x] Implement semantic search using NumPy
- [x] Compare cosine and Euclidean distances
- [x] Implement FAISS IndexFlatIP
- [x] Calculate Precision@K and Recall@K

Next:
- [ ] Compare NumPy and FAISS results
- [ ] Experiment with IndexFlatL2
- [ ] Benchmark 1K, 10K and 100K vectors
- [ ] Implement BM25 keyword retrieval
- [ ] Implement hybrid search with RRF
- [ ] Evaluate Recall@K, MRR and nDCG
- [ ] Experiment with HNSW
- [ ] Experiment with IVF and nprobe
- [ ] Add chunking and metadata
- [ ] Introduce reranking

---

## 16. Interview Takeaways

Q: What is FAISS?
A: A library for efficient dense-vector similarity search supporting exact and approximate nearest-neighbor algorithms.

Q: Is FAISS a vector database?
A: Not by itself. It provides indexing and search capabilities; applications must handle additional database and operational responsibilities.

Q: Why normalize embeddings?
A: To make dot product equivalent to cosine similarity and remove magnitude effects.

Q: Flat vs HNSW vs IVF?
A:
- Flat: Exact baseline
- HNSW: Graph-based fast approximate retrieval
- IVF: Cluster-based approximate retrieval
- IVF-PQ: Compression for memory efficiency

Q: Does FAISS improve semantic quality?
A: FAISS accelerates or organizes vector search. Embedding quality and retrieval strategy determine semantic relevance; ANN can introduce approximation errors.

Q: Which index should an architect choose?
A: Benchmark using dataset size, dimensions, memory, latency, throughput, update frequency, and required recall.

## Final Takeaway

Embeddings → Represent semantic meaning

Magnitude → Measures vector length

Normalization → Scales magnitude to 1

Dot Product → Measures alignment influenced by magnitude

Cosine Similarity → Measures directional alignment

FAISS → Performs efficient vector similarity search

Flat Index → Exact nearest-neighbor search

HNSW / IVF → Approximate search for scalability

Retrieval Evaluation → Measures relevance and ranking quality
