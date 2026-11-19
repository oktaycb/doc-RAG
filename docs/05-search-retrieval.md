# 05 — Search & Retrieval

Retrieval is the query-time half of RAG: turn a user question into the k chunks that best support an answer. The standard architecture has four stages — **transform → retrieve → fuse → rerank** — and the single most important empirical finding in 2024–2026 literature is that *stages stack*: hybrid retrieval and reranking each independently reduce failures, and their gains compose ([Anthropic: −49% → −67% top-20 failure reduction](https://www.anthropic.com/research/contextual-retrieval)).

## 1. Stage 0 — Query understanding (optional but cheap)

Raw user queries are often poor search queries. Transformations, in ascending order of cost:

- **Multi-query expansion**: ask an LLM to produce 3–5 paraphrases; retrieve for each; merge. Helps when users phrase things unusually. Cost: one small LLM call.
- **HyDE (Hypothetical Document Embeddings)**: have an LLM *imagine* an answer passage, embed *that*, and search with it. Often beats searching with the question, because answers resemble documents more than questions do. Risk: hallucinated hypotheticals on topics absent from the corpus.
- **Decomposition**: split compound questions ("compare X and Y on A, B, C") into sub-queries retrieved independently. Needed for multi-hop questions.
- **Step-back**: generalize the question ("what year did X file for bankruptcy" → "X's financial history") to retrieve context-bearing chunks.

**When to skip**: short factual lookups with clear terms, latency-critical paths. Add transformations only when your eval set shows retrieval misses that paraphrasing fixes ([07](07-evaluation.md) tells you).

## 2. Dense retrieval — strengths and blind spots

Dense (embedding cosine) search excels at meaning: paraphrases, cross-lingual matches, conceptual questions stated differently from the text. It reliably fails on:

| Query type | Why dense misses |
|---|---|
| Exact identifiers ("HMK 115", "AB-4721", error codes) | Semantically the code is just "some token"; lexical match is what matters |
| Rare proper nouns / new jargon | Possibly undertrained in the embedding model |
| Negation and subtle qualifiers ("not including X") | Embeddings blur negation |
| Very short queries over long passages | Vector compression collapses |

## 3. Sparse retrieval (BM25) — and why it refuses to die

BM25 scores by term frequency with saturation, inverse document frequency, and length normalization. It is 50+ years old and still wins benchmarks regularly — one 2026 benchmark found BM25 outperforming dense retrieval outright on its corpus ("From BM25 to Corrective RAG"). Its virtues: exact-term fidelity, zero training, robust zero-shot behavior, and perfect explainability (you can see *why* it matched).

**Turkish caveat**: BM25 matches surface strings, and Turkish morphology means the same lemma appears in many suffixed forms ("kitap", "kitabı", "kitapta"…). Stem both documents and queries (Zemberek roots or Snowball Turkish) or your sparse recall on Turkish text drops sharply. Alternatively, **BGE-M3's sparse output** gives you *learned* lexical weights that generalize across morphology without a stemmer ([03](03-embedding.md)).

**Contextual BM25**: index the context-enriched chunk text (Anthropic's variant) so the lexical index also carries chunk-level context. In Anthropic's ablation this was the step that took hybrid retrieval from −35% to −49% failures.

## 4. Hybrid retrieval + fusion

Run both retrievers, then merge the result lists. Two fusion methods:

### Reciprocal Rank Fusion (RRF) — the robust default

```
RRF_score(doc) = Σ_over_lists  1 / (k + rank_in_list)     k ≈ 60
```

No score normalization needed — it fuses *ranks*, not scores, which sidesteps the problem that cosine and BM25 scores live on incompatible scales. Give each list equal weight unless evaluation says otherwise. This is the standard choice.

### Weighted score fusion

Normalize each scorer (e.g., min-max or z-score over the candidate set), then blend: `score = α·dense + (1−α)·bm25`. More controllable, more fragile (normalization is unstable when a list is empty or scores cluster). Only bother when RRF demonstrably underperforms on your eval set.

Typical configuration: retrieve **top-50 from each** retriever → fuse → **top-150 candidates** → rerank → **top-20 to the generator** (Anthropic used exactly the 150→20 shape and found reranking worth the latency).

## 5. Reranking — the highest-ROI stage

The bi-encoder (embedding) scored query and document *separately*; a **cross-encoder** reads them *together* and outputs a relevance score. It is far more accurate — and far too slow to run over a whole corpus, which is why it lives at the top of the funnel on ~100–150 candidates only.

| Reranker | Where it runs | Cost | Notes |
|---|---|---|---|
| **bge-reranker-v2-m3** | Local (GPU or CPU) | Free | Multilingual, the standard local pick; pairs naturally with BGE-M3 |
| **Qwen3-Reranker (0.6B/4B/8B)** | Local | Free | Same family as Qwen3-Embedding; instruction-aware |
| **Cohere Rerank 3 / 3-lite** | API | $0.05 / $0.02 per 1M tokens (~$0.0025/request at 100 docs) | Strong multilingual; simplest to integrate |
| **Voyage rerank-3** | API | $0.05 per 1M tokens | Same family as voyage embeddings |
| **LLM-as-reranker** | API | Highest | Prompt an LLM to score/order candidates; flexible, slow, no dedicated model needed |

Latency reality: a local cross-encoder on CPU adds ~100–500ms for 100 candidates; API rerankers add one round trip (~200–400ms). If that's unacceptable, rerank fewer candidates (50→10) — most of the benefit survives.

**What it buys**: Anthropic's stacking measurement — reranking took their already-hybrid contextual system from 2.9% to 1.9% top-20 failure rate (−67% vs. baseline). In most personal-RAG builds, hybrid + rerank is *the* upgrade worth building before any exotic chunking ([02 §6](02-chunking.md)).

## 6. Small-to-big retrieval (parent documents)

Retrieve *small* chunks for precision, but return *bigger* units to the generator:

1. Index leaf chunks (200–400 tokens) with vectors + metadata pointing at parent section ids.
2. On a hit, load the parent section (800–2000 tokens) — deduplicate overlapping parents.
3. Feed parents (or parents + the original hit) to the generator within your context budget.

This decouples "what embeds well" from "what reads well" and fixes the chunk-too-small-for-context problem without weakening retrieval. It composes with hybrid and reranking.

## 7. Metadata filtering in the retrieval path

Before similarity search, apply hard filters from the query ("in the 2023 report", "only tables") → metadata predicates (date ranges, type, tags) → then ANN search within that subset. This is why [04 §3](04-indexing.md) insists on metadata discipline. An LLM query-planner step can extract filter predicates automatically; at personal scale a manual "filter chips" UI is often enough.

## 8. Measuring retrieval (preview of [07](07-evaluation.md))

Retrieval quality is measured *without* the generator, against a golden dataset mapping questions to known-relevant chunk ids:

- **Recall@k / hit rate**: fraction of queries where ≥1 (or all) relevant chunk appears in top-k. The workhorse metric; tune k with it.
- **MRR (Mean Reciprocal Rank)**: `mean(1 / rank_of_first_relevant)` — how high the *first* correct hit lands.
- **NDCG@k**: discounted gain over the ranked list — the metric when *multiple* chunks are relevant with graded relevance: `DCG = Σ rel_i / log2(i+1)`, normalized by the ideal ordering.

Plus RAGAS's **context precision** (are the top-ranked chunks actually relevant?) and **context recall** (does the retrieved set contain everything the reference answer needs?) for LLM-judged variants. Track retrieval metrics separately from generation metrics — the arXiv thesis study showed a good generator can mask a broken retriever, and you will never see that split unless you measure it.

## 9. Recommended progression

1. **Baseline**: dense-only, top-5, no transformations. (Walking skeleton — [08 Phase 2](08-project-roadmap.md).)
2. **+ Hybrid**: add BM25 (stemmed if Turkish) or BGE-M3 sparse, RRF fusion. Expect the single biggest failure-rate drop (−49% per Anthropic when combined with contextual enrichment).
3. **+ Rerank**: 150→20 with bge-reranker-v2-m3 or an API reranker. −67% cumulative.
4. **+ Contextual enrichment** of both indexes ([02 §2.7](02-chunking.md)).
5. **+ Query transformations** only where eval shows specific remaining miss classes.
Validate each step on your golden dataset before keeping it.

## Sources

- [Anthropic — Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval)
- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3) (dense+sparse+colbert, hybrid+rerank pipeline recommendation)
- [Qwen3-Embedding/Reranker card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Voyage pricing](https://docs.voyageai.com/docs/pricing) (rerank-3 $0.05/1M)
- [Claude Cookbook — contextual embeddings + reranking guide](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide)
- [TR-MTEB](https://github.com/selmanbaysan/mteb_tr) (Turkish retrieval evaluation tasks)
- HyDE: Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (arXiv 2212.10496)
