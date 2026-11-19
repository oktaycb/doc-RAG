# 02 — Chunking

Chunking is the decision of what a "retrievable unit" is. It is the highest-leverage preprocessing choice in RAG: it determines the granularity of everything downstream — what gets embedded, what gets indexed, what the retriever can find, and what the generator finally reads. Done badly, it caps your system's ceiling no matter how good your embedding model or reranker is.

## 1. Why chunking is hard: the context conundrum

Splitting a document into chunks solves two problems at once but creates a third:

- **Embedding quality**: a short, focused passage embeds more sharply than a whole page. Dense retrieval over long texts over-compresses semantics — the vector must average too many topics, so it matches everything weakly.
- **Generator budget**: you can only feed the model so many tokens per query, and precision beats volume.
- **Lost context (the cost)**: once split, a chunk no longer knows what document it came from. Anthropic calls this the "context conundrum": a chunk that says revenue "grew by 3%" is nearly useless for retrieval if the company name and quarter lived in a previous chunk. Pronouns, section titles, and document-level facts all evaporate at chunk boundaries.

Nearly every advanced chunking technique in this chapter is an attempt to recover that lost context while keeping chunks small.

## 2. The strategy landscape

### 2.1 Fixed-size (token) splitting

Cut the text every N tokens, optionally with an overlap window. Dead simple, predictable memory/cost, language-agnostic. Weakness: cuts mid-sentence and mid-thought, which corrupts both embeddings and retrieved readability.

> **Turkish note**: split by **tokens, not characters**. Agglutinative words mean one Turkish word often becomes multiple subword tokens; character-based splitting mangles words and skews chunk statistics. Count chunks in model tokens (the same tokenizer as your embedding model, ideally).

### 2.2 Recursive character splitting

Try to split on the strongest separator first (paragraph breaks `\n\n`), fall back to weaker ones (sentences, then words, then characters) until pieces fit within the size budget. This is the default in nearly every framework (`RecursiveCharacterTextSplitter` and equivalents). It respects natural boundaries while keeping sizes controlled — and the benchmark evidence below says it is a very strong default.

### 2.3 Structure-aware (document parsing) splitting

Parse the document's real structure — headings, list items, table rows, markdown/HTML elements — and split at structural boundaries, carrying the heading path (`Document > Section 3 > Subsection 3.2`) into each chunk's metadata. Best choice when your corpus is well-structured (manuals, reports, wikis). Tables deserve special handling: either keep a table whole as one chunk or serialize it to markdown/text with its caption.

### 2.4 Sentence/paragraph splitting

Split on sentence or paragraph boundaries, possibly grouping N sentences per chunk. Needs a sentence splitter; see the Turkish pitfalls in §5.

### 2.5 Semantic chunking

Embed individual sentences, then cut where the embedding similarity between adjacent sentences drops — a boundary between "topics." Variants:

- **Sliding-window (Kamradt-style)**: embedding discontinuity detection over a moving window.
- **Cluster-based**: group sentences by similarity (dynamic programming or linkage clustering) under a size cap.
- **LLM-driven**: ask an LLM to pick split points. Highest quality, wildly expensive.

### 2.6 Late chunking

Reverses the order of operations: instead of chunk-then-embed, you **embed-then-chunk**. Run the *entire document* through a long-context embedding model (8k+ tokens), producing contextual token embeddings; apply chunk boundaries afterward; mean-pool each chunk's tokens into its final vector. The paper describes it as "chunking applied after the transformer model and just before mean pooling." Because every token embedding already attended to the whole document, each chunk vector carries document-level context — the same goal as contextual retrieval, achieved architecturally instead of via an extra LLM call. Works with any sufficiently long-context embedding model without retraining ([arXiv 2409.04701](https://arxiv.org/abs/2409.04701)).

### 2.7 Contextual retrieval (contextual enrichment)

The other — and better-benchmarked — answer to context loss, from [Anthropic's research](https://www.anthropic.com/research/contextual-retrieval). Before indexing, ask an LLM (a small, cheap one is fine) to write a short siting note for each chunk: send it *the whole document plus the chunk*, and have it produce ~50–100 tokens placing the chunk in context ("This chunk is from Acme Corp's Q2 2024 earnings report, discussing…"). Prepend that note to the chunk, then embed and index it. Do the same for the BM25 side (**contextual BM25**). With prompt caching — load the document once, iterate over its chunks — Anthropic measured a one-time cost of **$1.02 per million document tokens**.

Measured effect on top-20 retrieval failure rate (1 − recall@20):

| Configuration | Failure rate | Reduction |
|---|---|---|
| Baseline (embeddings only) | 5.7% | — |
| + contextual embeddings | 3.7% | **−35%** |
| + contextual BM25 (hybrid) | 2.9% | **−49%** |
| + reranking | 1.9% | **−67%** |

The improvements stack: context, hybrid retrieval, and reranking each add independent gains. Chunk sizes of 5/10/20 retrieved were compared; top-20 won. In their experiments Gemini Text 004 and Voyage embeddings performed best, and custom contextualizer prompts (e.g., injecting a domain glossary) beat the generic prompt.

### 2.8 Parent-document / small-to-big retrieval

Index small chunks (sharp embeddings, precise retrieval) but *return their parent* — the surrounding section or a widened window — to the generator. You get retrieval precision and generation context simultaneously. This is an architectural pattern rather than a splitter, and it composes with everything above. Recommended default for report/manual-style corpora.

### 2.9 Proposition-based chunking

Decompose text into atomic self-contained statements ("propositions") and index those. Sharp recall for factoid questions, but loses narrative flow and multiplies chunk count. Niche; try only after the basics fail.

## 3. What the benchmarks actually say

The honest summary: **there is no universal winner, and "semantic" does not automatically beat "simple."**

[Chroma's research evaluation](https://www.trychroma.com/research/evaluating-chunking) (token-level precision/recall/IoU, 472 synthetic queries over ~328k tokens of State of the Union, Wikitext, chat logs, financial transcripts, and biomedical text; `text-embedding-3-large`, top-5 retrieval):

| Strategy (size) | Recall@5 |
|---|---|
| Recursive character 800/400 overlap | 85.4 |
| Recursive character 400 / 0 overlap | 89.5 |
| Token splitter | 87.0–87.9 |
| Kamradt semantic (defaults, ~660 tok) | 83.6 |
| Kamradt + length cap (~397 tok) | 87.1 |
| Cluster semantic, max 400 | **91.3** |
| LLM-driven chunker | **91.9** (at prohibitive cost — tens of minutes per corpus) |

Their conclusions, worth internalizing: strategies differ by up to ~9% recall; but plain recursive splitting, small (~200 tokens), with *zero* overlap, "often performs well in practice" when parameterized properly; the popular Kamradt semantic chunker scored *below* the recursive baseline at default settings, and its fix was a length cap rather than semantics itself; removing overlap improved boundary precision (IoU); and OpenAI Assistants' default (800 tokens + 400 overlap) produced the worst efficiency scores in their runs.

Contrast with an end-to-end industry benchmark (FloTorch, summarized in 2026 comparisons): semantic chunking scored ~54% answer accuracy — roughly 15 points *behind* recursive chunking. And a 2026 arXiv study ([2607.01852](https://arxiv.org/html/2607.01852v1)) on thirteen academic theses found cluster-based semantic chunking did not outperform recursive or fixed-size chunking while costing more compute.

Two further lessons from that same study, both underrated:

1. **Preprocessing artifacts poison retrieval.** Table-of-contents "dot leader" lines (`Introduction .......... 5`) matched metadata-style queries and flooded the top-k with garbage, driving context F1 to near zero *for every chunking strategy*. Clean your text before you blame your chunker.
2. **A good generator can mask a bad retriever.** On some questions the generator answered correctly from parametric knowledge despite failed retrieval — which means your *generation* quality can look fine while retrieval is broken. This is exactly why you measure retrieval separately (see [07](07-evaluation.md)).

## 4. Parameter guide

| Parameter | Recommendation | Notes |
|---|---|---|
| Chunk size | 200–500 tokens to start | 400 is a sane default; structure-aware corpora can go larger per section |
| Overlap | 0–15% | Benchmarks favor little/zero overlap for precision; overlap helps continuity but blurs boundaries |
| Unit | Tokens (model tokenizer) | Not characters; especially for Turkish |
| Boundaries | Paragraphs / headings / sentences | Never mid-sentence if avoidable |
| Metadata per chunk | `doc_id, title, section_path, page, chunk_index` | Cheap, invaluable for filtering + citation |
| Retrieved per query (k) | Start k=5; tune up to 20 with reranking | Anthropic: top-20 > top-10 > top-5 |

## 5. Turkish-language considerations

If your documents are Turkish (or mixed Turkish/English), chunking has language-specific traps:

- **Token inflation from agglutination.** Turkish suffix chains mean fewer words per token budget than English. Expect ~1.5–2x more tokens for equivalent content; size your chunks in tokens, and expect higher per-corpus embedding costs on cloud providers (billed per token).
- **Sentence splitting pitfalls.** Abbreviations ("Dr.", "vb.", "bkz."), decimals ("3.14"), and date formats ("12.03.2024") trigger false sentence boundaries in naive splitters. Use a splitter that handles Turkish abbreviations (e.g., a regex with a curated abbreviation list, or community NLP tooling like Zemberek-based tokenizers) rather than a blind `\. ` split.
- **Morphology dilutes BM25.** "ev", "evde", "evden", "evlerin" are different surface forms for one concept. If you use BM25 on Turkish text, apply a Turkish stemmer/lemmatizer (Zemberek roots, or snowball Turkish) on the indexing and query sides consistently — this materially improves the sparse half of hybrid retrieval (see [05](05-search-retrieval.md)).
- **Multilingual embeddings carry the load.** Your chunker's job is boundaries; retrieval semantics come from the embedding model. Choose a model with proven Turkish/multilingual strength ([03](03-embedding.md)) — that choice matters more than exotic chunking.

## 6. Decision guide

| Your situation | Start with |
|---|---|
| Generic prose, unknown structure | Recursive character, ~400 tokens, 0–10% overlap |
| Well-structured docs (manuals, reports, wikis) | Structure-aware split on headings + parent-document retrieval |
| PDFs with tables/forms | Layout-aware parsing first, then structure-aware splitting |
| Chunks that read ambiguously alone ("it", "this", unnamed entities) | Contextual retrieval enrichment (+ hybrid + rerank) |
| Very long documents, local-only stack | Late chunking with a long-context multilingual embedder, or parent-document pattern |
| Factoid lookups over dense reference material | Smaller chunks (~200) + propositions as a later experiment |

**The upgrade path that matters** (in order, each validated against your eval set per [07](07-evaluation.md)): recursive baseline → hybrid retrieval + rerank (biggest published win: −49% → −67% failures) → contextual enrichment → exotic chunking only if metrics still demand it.

## Sources

- [Anthropic — Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval)
- [Chroma Research — Evaluating Chunking Strategies for Retrieval](https://www.trychroma.com/research/evaluating-chunking)
- [arXiv 2607.01852 — Evaluating Chunking Strategies for RAG](https://arxiv.org/html/2607.01852v1)
- [arXiv 2409.04701 — Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models](https://arxiv.org/abs/2409.04701)
- [Atlan — Chunking Strategies for RAG (2026 guide)](https://atlan.com/know/chunking-strategies-rag)
- [Meilisearch — 13 RAG chunking strategies](https://www.meilisearch.com/blog/rag-chunking-strategies)
