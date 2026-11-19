# 01 — RAG Foundations

> This guide series covers everything you need to design, build, and measure a retrieval-augmented generation (RAG) system over your own documents: [chunking](02-chunking.md), [embedding](03-embedding.md), [indexing](04-indexing.md), [search & retrieval](05-search-retrieval.md), [citation](06-citation.md), and [evaluation](07-evaluation.md), culminating in a step-by-step [project roadmap](08-project-roadmap.md).

## 1. What RAG is and why it exists

Retrieval-augmented generation is a pattern where, instead of asking a language model to answer from its parametric memory, you first **retrieve** relevant passages from a knowledge base you control, then **generate** an answer grounded in those passages. The model's job changes from "know things" to "read and reason over what was just handed to it."

The pattern exists because of four problems that pure LLM usage cannot solve:

1. **Knowledge freshness** — a model only knows what was in its training data. Your documents are newer (or simply not public).
2. **Grounding** — generated answers drift toward plausibility. When the answer must come from a specific contract, report, or manual, plausibility is not good enough.
3. **Access boundary** — private documents were never in any training set.
4. **Cost and updating** — retraining or fine-tuning to add a document is absurd; adding it to an index is cheap and instant.

## 2. The full pipeline anatomy

A production RAG system is two pipelines: an **ingestion pipeline** that runs offline (or on document change), and a **query pipeline** that runs per user question. An **evaluation loop** sits around both.

```
INGESTION (offline)
┌─────────┐   ┌────────┐   ┌──────────┐   ┌────────────┐   ┌───────────┐
│ Ingest  │──▶│ Parse  │──▶│ Chunk    │──▶│ Contextual │──▶│ Embed     │
│ files   │   │ to text│   │          │   │ enrich     │   │           │
│ (PDF,   │   │ + meta │   │          │   │ (optional) │   │           │
│ docx…)  │   │        │   │          │   │            │   │           │
└─────────┘   └────────┘   └──────────┘   └────────────┘   └─────┬─────┘
                                                                 │
                                                        ┌────────▼────────┐
                                                        │ Index           │
                                                        │ (vectors + BM25 │
                                                        │  + metadata)    │
                                                        └─────────────────┘

QUERY (per question)
┌────────┐   ┌────────────┐   ┌──────────┐   ┌──────────┐   ┌───────────┐
│ Query  │──▶│ Transform  │──▶│ Retrieve │──▶│ Fuse     │──▶│ Rerank    │
│        │   │ (optional) │   │ top-N    │   │ (hybrid) │   │ top-k     │
└────────┘   └────────────┘   └──────────┘   └──────────┘   └─────┬─────┘
                                                                  │
┌────────────┐   ┌───────────┐   ┌────────────────┐              │
│ Evaluate   │◀──│ Cite      │◀──│ Generate       │◀─────────────┘
│ (offline + │   │ sources   │   │ grounded answer│
│  live)     │   │           │   │                │
└────────────┘   └───────────┘   └────────────────┘
```

Each block is a design decision with its own chapter:

| Block | What you decide | Chapter |
|---|---|---|
| Parse | How to extract text + structure + metadata from your file formats | 02, 08 |
| Chunk | How to split text into retrievable units | 02 |
| Enrich | Whether to prepend LLM-generated context to chunks | 02 |
| Embed | Which model turns text into vectors | 03 |
| Index | Where vectors, sparse terms, and metadata live | 04 |
| Transform | Whether to rewrite/expand the query | 05 |
| Retrieve/Fuse | Dense, sparse, or hybrid; how to combine | 05 |
| Rerank | Second-stage precision with a cross-encoder | 05 |
| Generate | Prompt design for grounded answers | 06 |
| Cite | How sources are attached to claims | 06 |
| Evaluate | Golden dataset, retrieval + generation metrics | 07 |

## 3. RAG vs. long context vs. fine-tuning

| | RAG | Long-context (stuff everything) | Fine-tuning |
|---|---|---|---|
| Knowledge updates | Re-index, seconds | Re-send, per request | Retrain |
| Corpus size limit | Millions of documents | Context window (~200k–1M tokens) | Effectively unbounded input |
| Cost per query | Retrieval + small prompt | Entire corpus billed every query | No per-token corpus cost |
| Verifiability / citations | Native (cite chunks) | Possible but unwieldy | None |
| Teaches behavior/style | No | No | Yes |
| Access control per user | Filter at retrieval | Hard | Hard |

A useful data point: Anthropic's contextual retrieval research notes that for knowledge bases under roughly **200,000 tokens (~500 pages)**, you may want to skip RAG entirely — prompt-cache the whole corpus and query it directly, which cut their latency by more than 2x and costs by up to 90%. RAG earns its complexity once your corpus grows past that, when you need citations, or when per-user access control matters. Fine-tuning is for *behavior* (tone, format, domain style), not for *facts* — it is a poor way to make a model know your documents and a terrible way to keep that knowledge current.

## 4. The ten design decisions

Before writing any code, you are implicitly making these decisions. Make them explicit:

1. **Corpus profile** — formats (PDF/DOCX/HTML/scans?), language(s) (Turkish? mixed?), size (pages, tokens), update frequency. This drives everything downstream.
2. **Deployment stance** — cloud APIs, fully local, or hybrid. Driven by privacy, budget, and hardware (see §5).
3. **Chunking strategy** — default recommendation: recursive character splitting at ~200–500 tokens with little or no overlap, then upgrade only if evaluation says so ([02](02-chunking.md)).
4. **Embedding model** — one model for the entire corpus; multilingual if your documents are not English ([03](03-embedding.md)).
5. **Index technology** — embedded (Chroma/FAISS/LanceDB) vs. server (Qdrant/pgvector/Elasticsearch) ([04](04-indexing.md)).
6. **Retrieval shape** — dense-only is a valid starting point; hybrid BM25+dense with reciprocal rank fusion plus a reranker is the standard upgrade ([05](05-search-retrieval.md)).
7. **Contextual enrichment** — prepend LLM-written context to chunks before indexing; largest single accuracy lever in published benchmarks, at a one-time indexing cost ([02](02-chunking.md)).
8. **Citation mechanism** — prompt-based markers (any LLM, local included) vs. structured citations (Claude API native feature) ([06](06-citation.md)).
9. **Evaluation harness** — a golden dataset of questions with known source chunks, and metrics you trust. Build this *before* tuning ([07](07-evaluation.md)).
10. **Refresh policy** — what happens when a document changes: re-chunk the whole document, upsert by stable `doc_id`, delete stale vectors ([04](04-indexing.md)).

## 5. Cloud vs. local: the framing decision

This guide is written stack-agnostic: every chapter presents both a **cloud path** (hosted APIs: Anthropic Claude for generation, OpenAI/Voyage/Cohere/Gemini for embeddings and rerankers) and a **local path** (Ollama/transformers with open-weight models: Qwen3-Embedding, BGE-M3, Llama/Qwen generators).

| Dimension | Cloud stack | Local stack (Ollama + open weights) |
|---|---|---|
| Retrieval quality ceiling | Highest (frontier embedding/rerank models) | Very good (Qwen3-Embedding-8B, BGE-M3 are near-SOTA) |
| Privacy | Documents leave your machine | Fully offline; nothing leaves |
| Cost | Per-token; cheap at hobby scale, real at corpus scale | Hardware electricity only |
| Latency control | Provider-dependent | Deterministic, local |
| Setup effort | API keys, minutes | Model downloads, VRAM/CPU budgeting |
| Rate limits | Yes | No |

A common and pragmatic pattern for personal-document RAG: **local embeddings** (BGE-M3 or Qwen3-Embedding-0.6B — fast, free, multilingual) + **cloud generation** (a strong frontier model for the final answer, where reasoning quality matters most), with the option to go fully local when privacy demands it.

## 6. What comes next

Read chapters in order for a full picture, or jump to what you need: [02 — Chunking](02-chunking.md) covers the highest-leverage preprocessing decisions; [07 — Evaluation](07-evaluation.md) explains how to know whether any of it worked; [08 — Project Roadmap](08-project-roadmap.md) sequences all of it into a concrete build plan.

## Sources

- [Anthropic — Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval)
- [Claude Docs — Citations](https://platform.claude.com/docs/en/build-with-claude/citations)
- [Chroma Research — Evaluating Chunking](https://www.trychroma.com/research/evaluating-chunking)
- [RAGAS docs — Available metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics)
