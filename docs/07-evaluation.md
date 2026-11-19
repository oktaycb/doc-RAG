# 07 — Evaluation: Measuring Your RAG System

This is the chapter that separates a RAG project from a demo. Without measurement you cannot tell whether semantic chunking helped, whether the reranker was worth its latency, whether the new embedding model is actually better *on your documents* — every change becomes vibes. The industry's honest confession: most published "X% improvement" claims are corpus-specific, and the 2026 academic literature repeatedly finds that techniques which win on one corpus lose on another. Your golden dataset is the only leaderboard that matters.

## 1. The golden dataset (build this first)

A golden dataset is a set of question/answer/source records that you treat as ground truth. Start with **30–100 rows**; quality beats quantity.

### Schema (keep it versioned, JSONL or CSV)

```
id:              q001
question:        "2023 raporunda KDV indirimi hangi bölümde düzenleniyor?"
reference_answer:"…"                       (ideal answer, used by reference-based metrics)
source_chunk_ids:[doc42::chunk_017]        (the chunk(s) that MUST be retrieved)
doc_ids:         [doc42]
tags:            [numeric, single-hop]
answerable:      true
```

### Where the questions come from

1. **Hand-written (30–50%)** — the questions you actually ask your documents. Include hard cases: questions needing two documents (multi-hop), numeric questions, paraphrased vocabulary (question words ≠ document words).
2. **LLM-synthesized (50–70%)** — for each of a sample of chunks, ask an LLM to write a question *whose answer is that chunk*; store `source_chunk_ids` for free. This mirrors the methodology Chroma used in their chunking benchmark (LLM-generated queries paired with verbatim excerpts, then filtered by cosine-similarity checks for duplicates and query–excerpt relevance). Filter aggressively: drop questions whose answer appears in many chunks (makes relevance ambiguous).
3. **Not-answerable rows (10–20%)** — questions that look plausible but have no answer in the corpus. The only way to measure refusal behavior ([06 §3](06-citation.md)).

Keep a small **held-out set** (10–15 rows) you never tune against; use it for final checks to avoid overfitting the system to your dev set.

## 2. Retrieval metrics (no LLM needed — run them constantly)

Given each question's `source_chunk_ids` and the retriever's top-k:

| Metric | Definition | Tells you |
|---|---|---|
| **Recall@k / hit rate** | fraction of queries with ≥1 relevant chunk in top-k (or: fraction of all relevant chunks found) | Can the retriever find the needle at all; choose k with it |
| **Precision@k** | relevant-in-top-k ÷ k | How polluted the context window is |
| **MRR** | mean over queries of `1/rank` of the first relevant hit | How high the *first* correct chunk lands |
| **NDCG@k** | `DCG@k / IDCG@k`, `DCG = Σ relᵢ / log₂(i+1)` | Ranking quality when several chunks are relevant with grades |

These are cheap, deterministic, and LLM-free — so run them on **every** experiment. A retrieval metric sheet per configuration (chunk size × embedding model × hybrid on/off × rerank on/off) is the core artifact of your project. Note the division of labor: recall@20 says whether the *candidate stage* works; MRR/NDCG@5 says whether the *rerank stage* orders well.

Also track: **median/p95 latency per query** and **cost per query** (embedding + rerank + generation tokens). Two systems with equal recall are separated by these.

## 3. Generation metrics (LLM-judged)

[RAGAS](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics) is the de-facto open-source standard; its core metrics map to failure modes:

| RAGAS metric | What it measures | Inputs | Failure mode it catches |
|---|---|---|---|
| **Faithfulness** | share of answer claims verifiable against retrieved contexts (LLM splits answer into claims, checks each; score = supported ÷ total) | question, answer, contexts | Hallucination beyond the sources |
| **Response relevancy** | pertinence to the question (LLM generates candidate questions *from the answer*; embedding cosine against the real question, averaged) | question, answer | Rambling, partial, or evasive answers |
| **Context precision** | whether relevant chunks rank above irrelevant ones (LLM labels each chunk, rank-weighted average precision) | question, contexts (opt. reference) | Good chunks buried under noise |
| **Context recall** | whether contexts contain everything in the reference (reference split into statements, each checked as attributable to contexts) | question, contexts, reference | Retrieval starvation → unanswerable from context |
| **Factual correctness** | statement-level TP/FP/FN vs. a reference answer → F1 | answer, reference | Wrong specifics despite fluent prose |
| **Noise sensitivity** | answer fidelity with noisy contexts vs. clean | all | Fragility to retrieval pollution |

Inputs in modern RAGAS pass through a sample object (`user_input`, `response`, `retrieved_contexts`, `reference`).

**Trust these numbers, but calibrate.** Documented cautions: an LLM-judge metric inherits the judge's biases (below), and the 2026 arXiv thesis study reported faithfulness computation *failing* (judge timeouts/invalid outputs) in 44% of their runs — treat any LLM-judged metric as noisy until you've eyeballed judge outputs on a sample of your own data and seen sensible judgments.

### Citation accuracy (custom but essential)

For citation-bearing answers ([06](06-citation.md)) measure: **% of factual sentences carrying ≥1 citation**, and **% of citations that actually support their sentence** (LLM judge: "does chunk X entail sentence Y?" — same machinery as post-hoc verification). This is the metric that catches the worst failure class: a confident answer whose citations point at the wrong place.

## 4. LLM-as-judge: the pitfalls

When an LLM grades your RAG's outputs, know its biases:

- **Position bias** — in pairwise comparisons, judges favor the first (or sometimes last) option; randomize order or use single-answer rubric scoring.
- **Verbosity bias** — longer answers score higher; counter with rubrics that reward brevity/conciseness explicitly.
- **Self-preference** — judging your generator's outputs with the *same* model inflates scores; use a different judge model when possible.
- **Rubric drift** — put the rubric in the prompt, keep it versioned, and never let "vibes" edits sneak in.
- **Calibration** — once, hand-label 20 outputs yourself and measure agreement with the judge. If Cohen's κ is poor, fix the rubric before trusting any trend.

## 5. The experiment protocol

The method that keeps the project honest:

1. **One variable at a time.** Change chunk size OR embedding model OR reranker — never two at once; otherwise attribution is impossible.
2. **Freeze the dataset version.** Every experiment runs against `golden-v3.jsonl`, recorded in the result row.
3. **Record configuration + metrics per run** (a simple results table/CSV): config hash, dataset version, recall@k, MRR, NDCG@k, RAGAS metrics (sampled — they cost LLM calls), latency, cost.
4. **Small-N discipline.** With 30–50 questions, don't quote p-values; look at effect sizes and consistency (did recall improve for most queries, or explode on three outliers?). Inspect per-query diffs, not just aggregates — the biggest insights live in "questions that regressed."
5. **Keep/regress decision** — adopt a change only if it improves the metric family it targets *without* regressing others beyond noise.

**Worked example** — "chunk size 400 vs 800": re-chunk, re-embed (two collections, [04 §5](04-indexing.md)), run both through identical retrieval+generation on golden-v3. Report recall@20, NDCG@5, faithfulness (on a 30-question sample), p95 latency. Keep whichever wins the *retrieval* metrics; the generator will follow.

## 6. Regression testing: evaluation as CI

Once the harness exists, evaluation becomes a *gate*, not a study:

- The golden suite runs automatically after any pipeline change (like unit tests).
- Define thresholds: e.g., recall@20 ≥ 0.90, faithfulness ≥ 0.85 on the sampled subset, refusal precision ≥ 0.80 on not-answerable rows.
- A change that drops any metric below threshold fails the "build" — revert or fix before proceeding.
- Re-run the full suite (including RAGAS) nightly or weekly if your corpus or models drift; snapshot results with timestamps to see long-run trends.

## 7. Tooling landscape

| Tool | Role |
|---|---|
| **RAGAS** | The metric library above; the default choice for the four core metrics |
| **DeepEval** | pytest-style RAG assertions; good if you think in unit tests |
| **TruLens / Arize Phoenix** | Tracing + feedback functions; Phoenix also does span-level observability of retrieval/generation |
| **LangSmith** | Hosted tracing, datasets, and eval runs in the LangChain ecosystem |
| **promptfoo** | Config-driven prompt/pipeline comparison matrices |
| **Custom harness** | JSONL golden set + your pipeline function + metric functions + a results table — often all you need at personal scale, and fully transparent |

Recommendation: start with a **custom harness** (you'll understand every number), adopt RAGAS for the LLM-judged metrics, add Phoenix or LangSmith when you need per-query tracing. Tracing (logging each query's transformed query, retrieved chunk ids+scores, generated answer, citations, latency, tokens) is not optional — every debugging session will live in those traces.

## 8. The metrics that matter most (summary)

If you track only five numbers: **recall@20** (retrieval ceiling), **NDCG@5** (ranking), **faithfulness** (groundedness), **refusal precision** (not-answerable discipline), and **p95 latency + cost** (usability). Everything else is refinement.

## Sources

- [RAGAS — Available metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics)
- [Chroma Research — Evaluating Chunking](https://www.trychroma.com/research/evaluating-chunking) (synthetic query methodology)
- [arXiv 2607.01852 — Evaluating Chunking Strategies for RAG](https://arxiv.org/html/2607.01852v1) (metric reliability caveats, parametric-knowledge masking)
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval)
- [DeepEval](https://github.com/confident-ai/deepeval) · [TruLens](https://www.trulens.org/) · [Arize Phoenix](https://github.com/Arize-ai/phoenix) · [LangSmith](https://www.langchain.com/langsmith) · [promptfoo](https://github.com/promptfoo/promptfoo)
