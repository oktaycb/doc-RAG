# RAG over Your Own Documents — A Research Guide

A detailed, source-backed reference for designing, building, and **measuring** a retrieval-augmented generation (RAG) system over personal documents (Turkish/English). Cloud APIs and fully-local open-source stacks are covered comparatively at every step. No code framework assumed — concepts, benchmarks, decision guides, and a build roadmap.

## Contents

| # | Chapter | What it covers |
|---|---|---|
| 01 | [RAG Foundations](docs/01-rag-foundations.md) | Pipeline anatomy (ingest → chunk → embed → index → retrieve → rerank → generate → cite → evaluate), RAG vs. long-context vs. fine-tuning, the ten design decisions, cloud vs. local framing |
| 02 | [Chunking](docs/02-chunking.md) | All strategies (recursive, structure-aware, semantic, late chunking, contextual retrieval, parent-document), benchmark evidence & contradictions, parameter guide, Turkish-language traps |
| 03 | [Embedding](docs/03-embedding.md) | 2026 model landscape (Qwen3, BGE-M3, OpenAI, Voyage, Cohere, Gemini) with prices/dimensions/contexts, how to read MTEB, TR-MTEB for Turkish, practical rules (MRL, prefixes, no model mixing) |
| 04 | [Indexing](docs/04-indexing.md) | Store comparison (Chroma/FAISS/Qdrant/pgvector/LanceDB), ANN algorithms (HNSW/IVF/quantization), metadata schema, BM25 index side, lifecycle (updates, model swaps, deletes) |
| 05 | [Search & Retrieval](docs/05-search-retrieval.md) | Dense vs. sparse failure modes, hybrid + RRF fusion, rerankers (local & API, with prices), query transformations (HyDE, multi-query), small-to-big retrieval, retrieval metrics |
| 06 | [Citation](docs/06-citation.md) | Prompt-based markers, Claude API native Citations (structured spans, location types), quote-extraction pattern, post-hoc NLI verification, source deep-linking, refusal design |
| 07 | [Evaluation](docs/07-evaluation.md) | Golden dataset construction, retrieval metrics (recall@k, MRR, NDCG — formulas), RAGAS metrics, LLM-as-judge pitfalls, citation accuracy, experiment protocol, regression testing, tooling |
| 08 | [Project Roadmap](docs/08-project-roadmap.md) | Phases 0–8 with entry/exit criteria, recommended default stacks (cloud & local), ten common mistakes |

## Suggested reading order

Building soon: **08 → 01 → 02 → 03 → 05 → 07**, skimming 04/06 as you reach those phases.
Reading for depth: 01 through 08 in order — each builds on the previous.

## The short version (if you read nothing else)

- **Chunking**: recursive character split, ~400 tokens, zero/small overlap. Don't adopt "semantic chunking" on faith — 2026 benchmarks show it losing to simple recursive on several corpora. The real win is **contextual enrichment** (−35% retrieval failures) + **hybrid BM25+dense** (−49%) + **reranking** (−67%) — effects that stack ([02](docs/02-chunking.md)).
- **Embedding**: one model, never mixed; multilingual for Turkish docs — Qwen3-Embedding (local, Apache-2.0, #1 MTEB-multilingual at release) or BGE-M3 (free dense+sparse in one pass); decide among finalists with TR-MTEB-informed evaluation on your own corpus ([03](docs/03-embedding.md)).
- **Index**: Chroma to start, Qdrant when you need native hybrid + filtered search; store rich metadata (`doc_id`, page, char offsets) from day one — citations depend on it ([04](docs/04-indexing.md), [06](docs/06-citation.md)).
- **Retrieval**: dense-only baseline → hybrid with RRF → cross-encoder rerank (150→20). Turkish BM25 needs stemming ([05](docs/05-search-retrieval.md)).
- **Citation**: Claude API native Citations if generating with Claude; prompt markers + post-hoc verification if fully local ([06](docs/06-citation.md)).
- **Measure**: build a 30–100-question golden dataset (including not-answerable questions) *before* tuning; track recall@20, NDCG@5, faithfulness, refusal precision, latency/cost; regression-test every change ([07](docs/07-evaluation.md)).

## Key sources

- [Anthropic — Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval)
- [Claude Docs — Citations](https://platform.claude.com/docs/en/build-with-claude/citations)
- [Chroma Research — Evaluating Chunking](https://www.trychroma.com/research/evaluating-chunking)
- [RAGAS — Available metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics)
- [Qwen3-Embedding model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) · [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [TR-MTEB (EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.471.pdf) · [mteb_tr tooling](https://github.com/selmanbaysan/mteb_tr)
- [Late Chunking (arXiv 2409.04701)](https://arxiv.org/abs/2409.04701) · [Evaluating Chunking for RAG (arXiv 2607.01852)](https://arxiv.org/html/2607.01852v1)
- [Voyage AI pricing](https://docs.voyageai.com/docs/pricing)

*Pricing and leaderboard positions shift quickly — figures are as of November 2026; verify against vendor pages before budgeting.*
