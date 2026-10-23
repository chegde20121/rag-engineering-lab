
# RAG Learning — BM25, Keyword Search & Lexical Retrieval

## Learning Objectives

Understand:
1. How search engines locate documents using inverted indexes.
2. How Term Frequency (TF) measures keyword occurrence.
3. How Inverse Document Frequency (IDF) measures term importance.
4. How BM25 ranks documents using relevance scoring.
5. When BM25 outperforms dense vector retrieval.
6. How BM25 fits into enterprise RAG architecture.

---

# 1. Inverted Index

## What is an Inverted Index?

An inverted index is a data structure that maps each term to the documents containing that term.

Instead of scanning every document for every query, a search engine looks up terms and retrieves matching document IDs.

### Example Documents

D1: "JES2 spool utilization is high"
D2: "WLM manages batch job performance"
D3: "JES2 spool capacity is exhausted"
D4: "TCPIP network connectivity failure"

### Forward Index

Document → Terms

D1 → [jes2, spool, utilization, high]
D2 → [wlm, manages, batch, job, performance]
D3 → [jes2, spool, capacity, exhausted]
D4 → [tcpip, network, connectivity, failure]

### Inverted Index

Term → Documents

| Term | Document IDs |
|------|--------------|
| jes2 | D1, D3 |
| spool | D1, D3 |
| batch | D2 |
| wlm | D2 |
| tcpip | D4 |

Query: "JES2 spool"

Lookup:
- jes2 → D1, D3
- spool → D1, D3
- Intersection → D1, D3

Only the matching documents need further scoring.

## Why Is It Useful?

- Efficient keyword lookup.
- Supports exact terms, identifiers and error codes.
- Avoids exhaustive document-text scanning.
- Forms the foundation of lexical search engines.

## Important Concepts

- Tokenization: Split text into searchable tokens.
- Normalization: Lowercase, Unicode handling, etc.
- Stopwords: Common words optionally excluded.
- Stemming: Reduce words to common stems.
- Posting list: Document IDs associated with a term.
- Term frequency: Often stored with posting information.
- Positions: Optional information for phrase/proximity search.

### Architect Considerations

For z/OS documentation, preserve identifiers such as:

- IEC030I
- B37-04
- SYS1.PARMLIB
- IEFBR14

Aggressive tokenization or punctuation removal may damage exact identifier matching.

---

# 2. Term Frequency (TF)

## What is TF?

Term Frequency measures how often a term appears in a document.

Example:

D1: "JES2 spool spool utilization"
D2: "JES2 spool utilization"

For the term "spool":

TF(spool, D1) = 2
TF(spool, D2) = 1

D1 contains "spool" more frequently.

## Basic Formula

TF(t,d) = Count of term t in document d

Other TF variants include:

Normalized TF = Term Count / Total Terms

## Limitation

More repetitions do not necessarily mean greater relevance.

Example:
- Document A mentions "spool" 3 times.
- Document B mentions "spool" 30 times.

Document B is not necessarily 10 times more relevant.

This motivates term-frequency saturation in BM25.

---

# 3. Inverse Document Frequency (IDF)

## What is IDF?

IDF measures how informative or rare a term is across a document collection.

Intuition:

- Common terms provide little discriminative information.
- Rare terms help identify specific documents.

Example:

Corpus: 100 documents

- "system" appears in 90 documents.
- "IEC030I" appears in 2 documents.

The term "IEC030I" is more discriminative.

## Classic IDF Formula

IDF(t) = log(N / df(t))

Where:

N = Total documents
df(t) = Number of documents containing term t

Example:

N = 100

For "system":

IDF = ln(100/90) ≈ 0.105

For "IEC030I":

IDF = ln(100/2) ≈ 3.912

Therefore:

IDF(IEC030I) > IDF(system)

## Key Points

- Rare terms receive higher IDF values.
- Common terms receive lower IDF values.
- IDF depends on the indexed corpus.
- Classic IDF differs from the IDF variant used in BM25.

---

# 4. TF-IDF

TF-IDF combines how frequently a term appears within a document and how informative that term is across the corpus.

## Formula

TF-IDF(t,d) = TF(t,d) × IDF(t)

Example:

Query term: "IEC030I"

If a document contains the term frequently and it is rare across the corpus, its TF-IDF weight becomes relatively high.

## Strengths

- Simple and interpretable.
- Useful for lexical relevance.
- Strong baseline for keyword search.

## Limitations

- Repeated terms can receive excessive weight with raw TF.
- Basic TF-IDF does not naturally account for document length.
- Cannot directly understand synonyms or paraphrased meaning.

BM25 addresses important ranking limitations of basic TF-IDF.

---

# 5. BM25 — Best Matching 25

## What Is BM25?

BM25, commonly Okapi BM25, is a probabilistic-retrieval-inspired ranking function used to score documents against keyword queries.

It combines:

1. Term frequency
2. Inverse document frequency
3. Term-frequency saturation
4. Document length normalization

BM25 scores documents based on lexical evidence, not semantic embeddings.

## BM25 Formula

BM25(D,Q) = Σ IDF(q) ×
[ TF(q,D) × (k1 + 1) ] /
[ TF(q,D) + k1 × (1 - b + b × |D|/avgdl) ]

The sum is calculated over query terms.

### Parameters

| Symbol | Meaning |
|--------|---------|
| TF(q,D) | Frequency of query term in document |
| IDF(q) | Importance of query term |
| k1 | Controls TF saturation |
| b | Controls document length normalization |
| \|D\| | Document length |
| avgdl | Average document length |

Common initial values:
- k1 = 1.2–2.0
- b = 0.75

These are starting points, not universal optimal settings.

## 5.1 Term-Frequency Saturation

BM25 recognizes that repeatedly mentioning a term should provide diminishing relevance gains.

Example:

| TF | Illustrative relevance contribution |
|----|-------------------------------------|
| 1 | Significant initial contribution |
| 2 | Additional contribution |
| 5 | Smaller incremental benefit |
| 20 | Approaches saturation |

Why?

Repeating "JES2" 50 times should not automatically make a document much more relevant than one mentioning it 5 times.

The `k1` parameter controls how rapidly TF saturates.

Smaller k1:
- Faster saturation.

Larger k1:
- More influence from repeated terms.

## 5.2 Document Length Normalization

Longer documents naturally contain more term occurrences.

BM25 compensates for this.

Example:

D1: 100 words, "spool" appears 4 times.
D2: 5,000 words, "spool" appears 4 times.

Assuming other factors are equal, BM25 with positive length normalization typically favors D1.

Parameter `b` controls normalization:

- b = 0: No document length normalization.
- b = 1: Full BM25 length normalization.
- b = 0.75: Common default.

## 5.3 BM25 IDF

A common nonnegative BM25 IDF variant is:

IDF(t) = ln(1 + (N - df(t) + 0.5)/(df(t) + 0.5))

Note:
BM25 implementations use different IDF variants and handling of frequent terms.

For example, `rank-bm25` uses an implementation-specific IDF treatment, so exact scores may differ between libraries.

## 5.4 How BM25 Ranks Documents

Query:
"JES2 spool exhausted"

Documents:

D1: "JES2 spool utilization is high"
D2: "WLM controls batch job scheduling"
D3: "JES2 spool capacity exhausted"

Expected intuition:

- D3 matches all three important query terms.
- D1 matches JES2 and spool.
- D2 has no matching query terms.

BM25 would typically rank:

1. D3
2. D1
3. D2

Actual scores depend on the full corpus, tokenization and BM25 parameters.

---

# 6. Python Implementation — BM25

Install:

```bash
pip install rank-bm25
```

## Example

```python
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd

documents = [
    "JES2 spool utilization is high",
    "WLM manages batch job performance",
    "JES2 spool capacity is exhausted",
    "TCPIP network connectivity failure"
]

# Simple tokenizer for learning purposes
tokenized_docs = [
    doc.lower().split()
    for doc in documents
]

bm25 = BM25Okapi(tokenized_docs)

def keyword_search(query, top_k=3):
    tokens = query.lower().split()

    scores = bm25.get_scores(tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    return pd.DataFrame([
        {
            "document_id": f"D{idx+1}",
            "score": round(float(scores[idx]), 4),
            "document": documents[idx]
        }
        for idx in top_indices
    ])

keyword_search("JES2 spool exhausted")
```

## Important

This uses a basic whitespace tokenizer for learning.

Production systems need more robust tokenization and identifier handling.

`BM25Okapi` uses corpus statistics such as:
- Document frequency
- Average document length
- Document term frequencies

Important practical caveat:
On tiny corpora, IDF handling can produce unexpected scores and ties. Expand the dataset and inspect scores before drawing conclusions.

---

# 7. BM25 vs Dense Vector Retrieval

## BM25 — Lexical Retrieval

BM25 uses exact token matching and statistical weighting.

Example:

Query:
"IEC030I B37-04"

A document containing those exact identifiers is likely to be ranked well.

Strengths:
- Exact identifiers
- Error codes
- Product names
- Technical terminology
- Explainable lexical matches

Limitations:
- Synonyms
- Paraphrasing
- Semantic relationships without shared words

## Dense Retrieval — Semantic Search

Dense retrieval converts queries and documents into embedding vectors.

Example:

Query:
"Why are batch jobs taking longer?"

Document:
"WLM resource contention increases workload execution time."

Even without identical words, embeddings may capture semantic relatedness.

Strengths:
- Natural-language meaning
- Paraphrased queries
- Related terminology
- Conceptual similarity

Limitations:
- Exact identifiers may be poorly represented.
- May retrieve semantically related but factually irrelevant passages.
- Requires embedding model inference.
- Dense similarity alone cannot guarantee relevance.

## Comparison

| Feature | BM25 | Dense Retrieval |
|---------|------|-----------------|
| Matching | Lexical | Semantic |
| Index | Inverted index | Vector index |
| Embedding model | Not required | Required |
| Exact error codes | Often strong | Model-dependent |
| Synonyms | Limited | Often strong |
| Paraphrases | Limited | Often strong |
| Explainability | Higher | Lower |
| Indexing cost | Generally lower | Embedding computation required |
| Ranking quality | Dataset-dependent | Dataset-dependent |

Neither method is universally better.

---

# 8. When to Use Which?

### Scenario 1 — Exact Error Code

Query:
"What does IEC030I indicate?"

Preferred:
BM25 or hybrid search.

Reason:
Exact identifier matching is highly valuable.

### Scenario 2 — Conceptual Question

Query:
"Why are my batch jobs running slowly?"

Preferred:
Dense or hybrid search.

Reason:
Relevant documents may describe WLM delays or resource contention without repeating query terms.

### Scenario 3 — Enterprise Technical RAG

Query:
"How can I resolve B37 abends caused by spool or storage constraints?"

Preferred:
Hybrid retrieval.

Reason:
The query contains both:
- Exact technical terminology
- Broader conceptual intent

---

# 9. Hybrid Search — Next Topic

Hybrid search combines lexical and semantic retrieval.

Pipeline:

User Query
   |
   +--> BM25 Retrieval ------+
   |                         |
   +--> Dense Retrieval -----+
                             |
                       Rank Fusion
                             |
                         Top-K Docs
                             |
                         Reranker
                             |
                            LLM

## Reciprocal Rank Fusion (RRF)

RRF combines rankings rather than comparing raw BM25 and cosine scores.

Formula:

RRF(d) = Σ 1 / (k + rank_i(d))

Where:
- rank_i(d) = Document rank in retrieval system i
- k = Rank smoothing constant, often 60

Why RRF?

BM25 and cosine scores use different scales.

RRF avoids directly combining incompatible raw score values.

---

# 10. Architect-Level Considerations

## Indexing
- Inverted index vs vector index
- Document updates and deletions
- Tokenizer selection
- Corpus statistics
- Index persistence

## Retrieval Quality
- Precision@K
- Recall@K
- MRR
- nDCG@K
- Golden query-document relevance dataset

## Performance
- Query latency
- Index size
- Memory utilization
- Retrieval throughput
- Index update frequency

## Enterprise Requirements
- Metadata filtering
- Tenant isolation
- Document-level authorization
- Exact identifiers and domain terminology
- Explainable ranking
- Search observability

Important:
ACL filtering must be enforced before unauthorized documents can be exposed, not merely by prompting the LLM to ignore them.

---

# 11. Hands-On Experiments

Use the same 10 documents from the embedding notebook.

- [ ] Build an inverted index manually.
- [ ] Calculate TF for a selected term.
- [ ] Calculate IDF for common and rare terms.
- [ ] Implement TF-IDF.
- [ ] Implement BM25 using rank-bm25.
- [ ] Compare BM25 vs FAISS Top-5 results.
- [ ] Test exact identifiers such as IEC030I.
- [ ] Test paraphrased queries.
- [ ] Calculate Precision@K and Recall@K.
- [ ] Compare retrieval latency.
- [ ] Implement RRF hybrid search.

Suggested notebook:
`03_bm25_keyword_search.ipynb`

---

# 12. Interview Questions

### Q1: What is an inverted index?

A data structure mapping terms to document postings, enabling efficient lexical retrieval without scanning all document text.

### Q2: What is TF?

Term Frequency measures how often a term appears in a document.

### Q3: What is IDF?

Inverse Document Frequency measures how discriminative a term is across the document collection.

### Q4: Why BM25 instead of basic TF-IDF?

BM25 introduces term-frequency saturation and document length normalization for more effective lexical ranking.

### Q5: What do k1 and b control?

- k1 controls TF saturation.
- b controls document length normalization.

### Q6: When is BM25 better than embeddings?

Often for exact identifiers, rare keywords, error codes and terminology that the embedding model may not represent reliably.

### Q7: Why use hybrid search?

To combine exact lexical matching with semantic understanding.

### Q8: Why not directly add BM25 and cosine scores?

Their scales are different and not inherently comparable. Rank fusion such as RRF avoids requiring raw score calibration.

---

# Key Takeaways

Inverted Index → Maps terms to documents.

TF → Measures frequency inside a document.

IDF → Measures rarity across documents.

TF-IDF → Combines frequency and rarity.

BM25 → Adds TF saturation and length normalization.

Dense Retrieval → Matches semantic meaning using embeddings.

Hybrid Search → Combines lexical and semantic retrieval.

RRF → Fuses ranked retrieval results.

Retrieval Evaluation → Measures ranking quality and relevance.

## Final Understanding

BM25 answers:
"Which documents contain the important query terms?"

Dense retrieval answers:
"Which documents are semantically close to the query?"

Hybrid retrieval combines both approaches to improve coverage across lexical and semantic queries.

For enterprise RAG, especially technical documentation with exact identifiers and natural-language questions, both methods should be evaluated using representative workloads.
