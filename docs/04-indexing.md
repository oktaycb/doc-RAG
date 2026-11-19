# 04 — Indexing

The index is where your embedded chunks live so they can be found in milliseconds. "Indexing" covers three things that must be designed together: **where** the data lives (the store), **how** nearest-neighbor search runs (the ANN algorithm), and **how the collection stays correct over time** (lifecycle: updates, model changes, deletes).

## 1. Choosing a vector store

For a personal-document RAG the honest answer is: any of the first four will serve you well; pick by the shape of the rest of your project.

| Store | Shape | Hybrid (dense+sparse) | Metadata filtering | Setup | Sweet spot |
|---|---|---|---|---|---|
| **Chroma** | Embedded library (in-process, persistent) | Via built-in/leaky abstractions | Yes | Trivial | Fastest path from zero; the "walking skeleton" choice |
| **FAISS** | Library, not a database | No (bring your own BM25) | None (external bookkeeping) | Trivial to start, manual persistence | Raw ANN speed experiments; you build the DB around it |
| **Qdrant** | Client-server (also embedded mode) | Yes (native sparse vectors, BGE-M3 pairings) | Strong, filterable payloads | One container/file | Best default once you outgrow embedded stores; quantization built in |
| **pgvector** | Postgres extension | Partial (combined with FTS) | Full SQL power | Existing Postgres | If your documents already live in a DB; joins with real tables |
| **LanceDB** | Embedded, disk-based columnar | Yes | Yes | Trivial | Large local corpora, zero servers |
| **Milvus / Elasticsearch / OpenSearch** | Server clusters | Yes (ES/OpenSearch natively) | Yes | Heavier | Scale-out or when you already run Lucene stack |

Decision shortcuts: solo project, < ~1M chunks, no existing infra → **Chroma or LanceDB**; expect hybrid search and payload filtering to matter → **Qdrant**; corporate/SQL environment → **pgvector**; hundreds of millions of vectors or existing Elastic → the big servers.

## 2. How nearest-neighbor search actually works

Exact (brute-force) comparison of a query against every vector is fine up to roughly 100k–1M vectors with modern BLAS; below that, skip ANN entirely and use flat exact search — it is simplest and perfectly fast. Beyond that, approximate nearest neighbor (ANN) indexes trade a little recall for a lot of speed.

### HNSW (hierarchical navigable small world) — the default

A layered proximity graph: search descends through sparse upper layers, then fine-searches in the dense bottom layer.

- `M` — edges kept per node (16 is the common default; 32–48 improves recall at memory cost).
- `ef_construction` — build-time search width; higher = better graph, slower build (128–200 typical).
- `ef_search` — query-time search width; **the recall/speed dial you tune after deployment**. Rule: `ef_search ≥ k`, raise it until recall on your eval set stops improving.
- Costs: memory-hungry (graph edges + vectors in RAM), excellent latency (sub-10ms at millions of vectors).

### IVF (inverted file index) — cluster-based

K-means the corpus into `nlist` clusters; at query time scan only the `nprobe` nearest clusters. Needs a training pass (sample of your vectors). `nprobe` is the recall/speed dial. Cheaper memory than HNSW, slightly lower ceiling on recall/latency trade.

### Quantization — the storage lever

Scalar quantization (float32→int8, 4x smaller), product quantization (PQ, 8–32x smaller, more recall loss), binary quantization (32x, works well with rescoring). Relevant the moment your corpus reaches tens of millions of vectors; unnecessary for personal scale.

**Practical guidance**: personal scale (≤ ~500k chunks) → flat or HNSW, default parameters, spend your effort on evaluation instead of index tuning. ANN parameters are the *last* thing to optimize, and only if latency actually bothers you.

## 3. Metadata: the half of the index people forget

Every chunk needs a stable identity and rich metadata, because filtering and citation both depend on it:

```
chunk_id:        stable unique id (e.g., doc_id::chunk_0007)
doc_id:          stable document identifier (file hash or managed id)
title, section_path, page (or page range), char_start/char_end
chunk_index:     position within document
source_uri:      path/URL for click-through
created_at, updated_at
embedding_model: model name+version that produced the vector
custom:          type (table/prose/heading), tags, dates from content
```

Two rules:

1. **`doc_id` + `chunk_index` must survive re-ingestion** of the same document, so updates can upsert instead of duplicating. Derive `doc_id` from a content hash, not from a mutable filename.
2. **Store `embedding_model`** (or name the collection after it): the day you switch embedding models, you must know exactly which vectors to invalidate ([§5](#5-lifecycle-updates-model-changes-and-deletes)).

Filtering at retrieval time (date ranges, document type, tags) is what turns a toy into a usable system — Qdrant/pgvector/ES support filtered ANN natively; with FAISS or Chroma you pre- or post-filter by metadata yourself.

## 4. The sparse (BM25) side of the index

If you plan hybrid retrieval ([05](05-search-retrieval.md)) — and you should — you need a lexical index alongside vectors:

- **Inverted index options**: SQLite FTS5 (embedded, zero-ops), Tantivy (Rust, excellent), Elasticsearch/OpenSearch (if already deployed), or in-memory BM25 libraries for small corpora.
- **Turkish preprocessing here**: BM25 matches surface forms, so apply consistent Turkish stemming (Zemberek roots or Snowball Turkish) to both documents and queries, or accept recall loss on suffixed forms ([02 §5](02-chunking.md)).
- **Contextual BM25**: index the *context-enriched* chunk text (Anthropic's variant) so exact-term search benefits from the added context too.
- Keep term statistics (df/idf) computed over the same chunk set as the vector side, so fusion scores compare compatible distributions.

## 5. Lifecycle: updates, model changes, and deletes

**Incremental document update** (a file changed): treat each document atomically — delete all chunks where `doc_id = X`, re-chunk and re-embed the new version, insert with the same `doc_id`, refresh the BM25 index for that document. Never diff-chunk partially: chunk boundaries are global to a document.

**Model change** (new embedding model): you cannot mix vectors from two models in one space ([03 §4](03-embedding.md)). Either (a) build a *new collection* named `corpus-v2-modelname`, evaluate side-by-side, then swap the alias/pointer — the zero-downtime pattern — or (b) re-embed in place with downtime. Option (a) also gives you A/B data from your eval harness for free.

**Deletes**: remove from vector index + sparse index + metadata store together; a `deleted` flag without actual removal eventually leaks stale chunks into answers. If documents carry legal sensitivity, actual deletion matters (GDPR-style), so prefer stores that support true deletes.

**Versioning**: keep `(chunk content hash → chunk_id)` so re-runs are idempotent; keep the ingestion pipeline version next to the collection so you can reason about "which chunks are stale."

## 6. Sizing example

1,000 documents averaging 20 pages (~10k tokens each) ≈ 10M tokens ≈ **~25k chunks** at 400-token chunks. At 1024 dims float32: vectors ≈ 100 MB; HNSW graph overhead roughly 1–2x the raw vectors; metadata + text in SQLite: a few hundred MB. All of this fits in RAM on a laptop — confirming that at personal scale your energy belongs in chunking, embedding choice, and evaluation, not index engineering.

## Sources

- [Chroma](https://www.trychroma.com/) · [FAISS](https://github.com/facebookresearch/faiss) · [Qdrant](https://qdrant.tech/) · [pgvector](https://github.com/pgvector/pgvector) · [LanceDB](https://lancedb.com/)
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval) (contextual BM25, collection hygiene practices)
- HNSW: Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*; IVF/PQ: standard FAISS documentation
