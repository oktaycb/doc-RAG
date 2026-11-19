# 03 — Embedding

An embedding model maps text into a fixed-size vector such that semantically similar texts land close together in that space. In a RAG system it is the semantic lens of your retriever: it converts both your indexed chunks and the user's query into the same vector space, and similarity search finds the nearest chunks. Everything about your corpus — its language, its vocabulary, its typical passage length — must survive the trip through this one model.

## 1. Minimal theory (what you actually need)

- **Bi-encoder vs. cross-encoder.** An embedding model is a *bi-encoder*: query and document are encoded **independently** into vectors, compared by cosine similarity. This is what makes search fast (precompute document vectors once). A *cross-encoder* (reranker, see [05](05-search-retrieval.md)) reads query+document *together* and is far more accurate per pair but cannot be precomputed — that's why it's used only on the small candidate set.
- **Cosine similarity** is the standard metric: the cosine of the angle between vectors, ignoring magnitude. Normalize vectors (unit length) and cosine becomes a dot product.
- **Dimensionality** ranges from ~384 (small local models) to 4096 (large open models). Higher isn't automatically better; it trades storage and speed for a little expressiveness. Storage math: a 1024-dim float32 vector ≈ 4 KB; 1M chunks ≈ 4 GB before metadata and index overhead.
- **Asymmetry**: some models are trained to treat queries and passages differently (see instruction prefixes, §4).

## 2. How to read MTEB (the leaderboard that matters)

[MTEB](https://huggingface.co/spaces/mteb/leaderboard) (Massive Text Embedding Benchmark) aggregates dozens of tasks. Two rules for RAG use:

1. **Look at the retrieval column, not the overall average.** Overall scores blend classification, clustering, and STS tasks that don't reflect your use case. Retrieval-task score is the one that predicts RAG performance.
2. **Leaderboards are a shortlist, not a decision.** Their corpora are Wikipedia-based English/multilingual sets — not your documents. The final ranking must come from your own golden-dataset evaluation ([07](07-evaluation.md)).

For Turkish documents specifically, check [TR-MTEB](https://aclanthology.org/2025.findings-emnlp.471.pdf) — the first large-scale Turkish embedding benchmark (26 datasets across six task types including retrieval; tooling at [github.com/selmanbaysan/mteb_tr](https://github.com/selmanbaysan/mteb_tr) and a [Hugging Face org](https://huggingface.co/trmteb)). It exists precisely because multilingual averages hide Turkish-specific rankings; where two models tie on the multilingual board, TR-MTEB retrieval results are the tiebreaker. (Newer Turkish-specialized models also appear in the literature, e.g. [TurkEmbed](https://arxiv.org/html/2511.08376v1), worth tracking.)

## 3. The 2026 model landscape

### Cloud / API models

| Model | Price ($/1M tokens) | Dimensions | Context | Notes |
|---|---|---|---|---|
| OpenAI text-embedding-3-small | $0.02 | 1536 | 8,191 | The budget default; solid English, decent multilingual |
| OpenAI text-embedding-3-large | ~$0.065 (post-2025 cut) | 3072 (MRL-trimmable) | 8,191 | Verify current pricing; supports dimension reduction |
| Voyage voyage-4-large | $0.12 | up to 2048* | 32k* | Top-tier retrieval family; 200M free tokens |
| Voyage voyage-4 | $0.06 | up to 1024* | 32k* | Balanced flagship |
| Voyage voyage-4-lite | $0.02 | up to 512* | 32k* | Cheap tier |
| Cohere Embed v4 | $0.12 text / $0.47 image | flexible (MRL-style) | 128k | Multimodal (text+images); compresses long docs |
| Gemini Embedding (2nd gen) | see Google pricing | 3072 (MRL) | 8k+ | Reported near the top of retrieval benchmarks (67.7 MTEB retrieval per aggregator guides) |

\* check the vendor's model card for current per-tier dimension/context specifics; pricing confirmed from [Voyage pricing](https://docs.voyageai.com/docs/pricing) (Nov 2026); OpenAI figures cross-checked against 2026 pricing pages; Cohere v4 figures are from aggregator pages and should be verified against Cohere docs before budgeting.

### Local / open-weight models (run via Ollama, transformers, vLLM)

| Model | Params | Dimensions (MRL) | Context | License | Why it matters |
|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | 0.6B | 1024 (32–1024) | 32k | Apache-2.0 | Best quality-per-GB; instruction-aware |
| Qwen3-Embedding-4B | 4B | 2560 | 32k | Apache-2.0 | Mid tier |
| Qwen3-Embedding-8B | 8B | 4096 | 32k | Apache-2.0 | **70.58 MTEB-Multilingual — #1 on the multilingual leaderboard at release (Jun 2025)** |
| BGE-M3 | ~0.6B-class | 1024 | 8,192 | MIT | Outputs **dense + sparse (learned lexical weights) + ColBERT-style multi-vector in a single forward pass**; 100+ languages |
| multilingual-e5-large | ~0.6B-class | 1024 | 512 | MIT | Aging but proven; requires "query:"/"passage:" prefixes |
| EmbeddingGemma (and similar small models) | <1B | ~768 (MRL) | 2k+ | open | Ultra-cheap local option for CPU-only setups |

Qwen3-Embedding card facts (from [the model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)): all three sizes are instruction-aware (adding a task instruction to the *query* yields a typical 1–5% gain; write instructions in English even for Turkish content), support user-defined output dimensions (Matryoshka), 32k context, and 100+ languages. BGE-M3 ([card](https://huggingface.co/BAAI/bge-m3)) needs no instruction prefix and explicitly recommends a hybrid (dense+sparse) + reranker pipeline — it hands you both halves of hybrid search in one model.

## 4. Practical rules (the ones people violate)

1. **Never mix embedding models** between index and query. Vectors from different models live in different spaces; cosine between them is noise. Changing models means re-embedding the entire corpus (see [04 §5](04-indexing.md)).
2. **Respect instruction prefixes.** e5-family models require `"query: …"` and `"passage: …"` prefixes. Qwen3 expects a task instruction on queries. Omitting these silently costs a few retrieval points.
3. **Use Matryoshka (MRL) support when storage-bound.** Qwen3 (32–1024 dims on the 0.6B) and OpenAI 3-large (`dimensions` parameter) let you truncate vectors with minimal quality loss — halve storage before you buy a smaller model.
4. **Normalize vectors** unless your store requires otherwise; it makes cosine equivalent to dot product and keeps score thresholds stable across models.
5. **Embed the contextualized chunk text**, not the raw one, if you adopt contextual retrieval ([02 §2.7](02-chunking.md)) — the enrichment only helps retrieval if the model sees it.
6. **Benchmark on your own corpus.** Take the leaderboard top-5 as candidates, embed your golden dataset's source chunks, measure Recall@k/MRR ([07](07-evaluation.md)), and let your Turkish documents pick the winner.

## 5. Cloud vs. local, concretely

| | Cloud API | Local (Ollama / transformers) |
|---|---|---|
| Quality ceiling | Frontier models (Gemini/Voyage/Qwen3-8B territory) | Qwen3-0.6B/BGE-M3 within a few points of frontier on multilingual retrieval |
| Privacy | Text leaves your machine | Fully offline |
| Cost | Per token; a 1M-token corpus ≈ $20 with 3-small (one-time), plus re-embeds on model change | Hardware only |
| Throughput | Provider rate limits | Your CPU/GPU; batching local |
| Ops | API keys, outages | Model files (~0.6–16 GB), version pinning |

**Default recommendation for a personal Turkish/English document corpus:** start local with **Qwen3-Embedding-0.6B** (or **BGE-M3** if you want free hybrid dense+sparse signals out of the box), measured against **bge-m3**-style baselines via TR-MTEB-informed selection; upgrade to a cloud frontier embedder only if your eval harness shows a retrieval-quality gap.

## 6. Decision guide

| Situation | Pick |
|---|---|
| Privacy first, fully offline | BGE-M3 or Qwen3-Embedding-0.6B (Apache/MIT, multilingual) |
| Maximum retrieval quality, cost secondary | Gemini Embedding or voyage-4-large; verify on your corpus |
| Cheapest cloud path | text-embedding-3-small ($0.02/1M) or voyage-4-lite |
| Long documents per chunk (32k) | Qwen3-Embedding family |
| Free hybrid dense+sparse from one model | BGE-M3 |
| Turkish is primary language | Shortlist via multilingual boards, decide via TR-MTEB retrieval results + own golden set |

## Sources

- [Qwen3-Embedding-0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3)
- [Voyage AI pricing](https://docs.voyageai.com/docs/pricing)
- [TR-MTEB paper (EMNLP 2025 Findings)](https://aclanthology.org/2025.findings-emnlp.471.pdf) · [tooling](https://github.com/selmanbaysan/mteb_tr) · [HF org](https://huggingface.co/trmteb)
- [TurkEmbed (arXiv 2511.08376)](https://arxiv.org/html/2511.08376v1)
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [2026 embedding model comparison guides](https://www.respan.ai/articles/best-embedding-models) · [MTEB scores overview](https://app.ailog.fr/en/blog/guides/choosing-embedding-models)
