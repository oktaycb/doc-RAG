# 08 — Project Roadmap: Building Your RAG System

This chapter sequences everything from [01](01-rag-foundations.md)–[07](07-evaluation.md) into a build plan for a personal-document RAG system. The ordering principle: **get a walking skeleton running, then measure, then upgrade one thing at a time** — evaluation infrastructure arrives *early*, because every later decision depends on it.

Assumed context: your own documents (likely Turkish, possibly mixed Turkish/English), personal scale (hundreds to thousands of files), no team. Time estimates are for focused solo work.

---

## Phase 0 — Scope definition (½ day)

**Do:** Inventory the corpus — file formats (PDF? DOCX? scans?), count + total pages/tokens, language mix, update frequency. Write down the 10 questions you most want to ask your documents. Decide the privacy stance: cloud APIs allowed, or fully local ([01 §5](01-rag-foundations.md)).

**Exit criteria:** a one-page scope note. If total corpus < ~200k tokens (~500 pages), reconsider: Anthropic's research suggests prompt-caching the whole corpus instead of RAG at that scale.

## Phase 1 — Parsing & corpus hygiene (1–2 days)

**Do:** Convert every file to clean text with extracted metadata (title, page mapping, dates). PDFs with layout/tables need a layout-aware parser; keep page↔text alignment for later citations ([06 §2](06-citation.md)). Clean aggressively: drop table-of-contents dot-leader lines, headers/footers, duplicate boilerplate — the arXiv chunking study showed TOC artifacts alone drove context-F1 to near zero across *all* chunking strategies ([02 §3](02-chunking.md)).

**Exit criteria:** a directory of parsed documents + a metadata manifest (`doc_id, title, source_path, page_offsets`); spot-check five documents against their originals.

## Phase 2 — Walking skeleton (1 day)

**Do:** The simplest end-to-end pipeline: recursive chunking ~400 tokens / 0–10% overlap ([02 §4](02-chunking.md)), one local multilingual embedder — **Qwen3-Embedding-0.6B** or **BGE-M3** ([03](03-embedding.md)) — into **Chroma** with full metadata ([04 §3](04-indexing.md)), dense retrieval top-5, one grounded prompt with citation markers + refusal rule ([06 §1.1](06-citation.md)). Generator: cloud frontier model, or a local model if privacy demands.

**Exit criteria:** you can ask a question and get a cited answer in <10 s. It will be mediocre. That's fine — it's the baseline everything is measured against.

## Phase 3 — Golden dataset v1 (1 day, before ANY tuning)

**Do:** 30–100 rows per the schema in [07 §1](07-evaluation.md): hand-written hard questions, LLM-synthesized per-chunk questions with `source_chunk_ids`, and 10–20% **not-answerable** rows. Split off a small held-out set.

**Exit criteria:** `golden-v1.jsonl` frozen. Skipping this phase is the single most common project-killing mistake — without it, Phases 4–7 become guesswork.

## Phase 4 — Evaluation harness + baseline numbers (1 day)

**Do:** Implement the no-LLM retrieval metrics — recall@k, MRR, NDCG@k ([07 §2](07-evaluation.md)) — against the golden set; run the baseline; add a results table capturing config + metrics per run; start per-query tracing (log transformed query, retrieved ids+scores, answer, latency, tokens).

**Exit criteria:** a one-command eval run producing baseline numbers. Expect the baseline to be humbling; that's the point.

## Phase 5 — Hybrid retrieval + reranking (2–3 days)

**Do:** Add the sparse side: BM25 (with Turkish stemming — Zemberek/Snowball — applied consistently to both sides) or BGE-M3's native sparse weights ([05 §3](05-search-retrieval.md)). Fuse with RRF (top-50 per retriever → top-150 candidates). Add a cross-encoder reranker: `bge-reranker-v2-m3` (local) or Cohere/Voyage rerank API ([05 §5](05-search-retrieval.md)), 150→20. 

**Exit criteria:** recall@20 and NDCG@5 improved on golden-v1; latency still acceptable; numbers recorded. This is typically the largest single quality jump (Anthropic: hybrid −49%, +rerank −67% top-20 failures).

## Phase 6 — Contextual enrichment (1–2 days, optional but high-value)

**Do:** If chunks read ambiguously in isolation, generate a 50–100-token context note per chunk with a cheap model and prepend before embedding + BM25 indexing ([02 §2.7](02-chunking.md)). Cost ≈ $1.02 per million document tokens with prompt caching on Claude Haiku-class models; estimate yours upfront. Local alternative: try **late chunking** with your long-context embedder ([02 §2.6](02-chunking.md)).

**Exit criteria:** failure-rate reduction measurable against Phase 5 numbers (Anthropic measured −35% from enrichment alone, pre-hybrid).

## Phase 7 — Citations done properly (1–2 days)

**Do:** If generating with Claude API: switch to native Citations — document blocks + `citations: {"enabled": true}`, custom-content blocks mapping back to your chunk ids ([06 §1.2](06-citation.md)). If fully local: keep prompt markers and add post-hoc verification (overlap/NLI check per cited sentence) ([06 §1.4](06-citation.md)). Build the source-map UI: click citation → open file at page ([06 §2](06-citation.md)).

**Exit criteria:** every factual sentence carries a citation; clicking it lands on the right page; citation accuracy measured ([07 §3](07-evaluation.md)).

## Phase 8 — Full evaluation suite + regression gate (1–2 days, then ongoing)

**Do:** Add the LLM-judged metrics (RAGAS faithfulness, response relevancy, context precision/recall — sampled, with judge-bias controls from [07 §4](07-evaluation.md)); compute refusal precision on the not-answerable rows; set regression thresholds and wire the suite to run after every pipeline change ([07 §6](07-evaluation.md)).

**Exit criteria:** any pipeline change that degrades metrics is caught automatically. From here, the project is a loop: inspect traces → find a failure class → fix the one thing → re-run the suite.

## Recommended default stacks

| Component | Cloud-friendly default | Fully-local default |
|---|---|---|
| Chunking | Recursive 400 tok | same |
| Embedding | Qwen3-Embedding-0.6B local, or voyage-4/text-embedding-3-small | Qwen3-Embedding-0.6B / BGE-M3 |
| Sparse index | BM25 + Turkish stemmer (or BGE-M3 sparse) | same |
| Store | Chroma → Qdrant when it grows | same |
| Reranker | bge-reranker-v2-m3 local, or Cohere rerank-3 | bge-reranker-v2-m3 |
| Generator | Claude (Sonnet class) with native Citations | Ollama (Qwen/Llama class) + prompt citations |
| Eval | Custom harness + RAGAS | same |

## Ten common mistakes (in the order people make them)

1. **Tuning before building the golden dataset** — every decision is vibes (Phase 3 exists for a reason).
2. **Changing two variables per experiment** — attribution becomes impossible ([07 §5](07-evaluation.md)).
3. **Trusting MTEB over your own corpus** — leaderboards shortlist; your golden set decides ([03 §2](03-embedding.md)).
4. **Blaming the chunker for parsing artifacts** — clean TOC/headers/footers first ([02 §3](02-chunking.md)).
5. **Character-based chunk sizes on Turkish** — token-based sizing, always ([02 §5](02-chunking.md)).
6. **Mixing embedding models without full re-index** — vectors from different models are incomparable ([03 §4](03-embedding.md)).
7. **Stuffing top-20 chunks into the prompt raw** — rerank and deduplicate parents first ([05 §5–6](05-search-retrieval.md)).
8. **Never testing refusal** — the not-answerable rows catch confident nonsense ([06 §3](06-citation.md)).
9. **Treating citations as decoration** — measure citation accuracy or don't claim verifiability ([07 §3](07-evaluation.md)).
10. **Adopting semantic/exotic chunking before hybrid+rerank** — the published evidence favors the boring stack first; semantic chunking loses to recursive on multiple 2026 benchmarks ([02 §3](02-chunking.md)).

## The loop

After Phase 8 the project becomes iterative: the eval harness tells you which failure class dominates (retrieval misses → [05](05-search-retrieval.md); faithfulness → [06](06-citation.md); latency → rerank fewer candidates or smaller chunks; cost → MRL-truncate embeddings or downsize the generator). One change per cycle, measured every time.
