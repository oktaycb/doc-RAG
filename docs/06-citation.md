# 06 — Citation

Citations connect every claim in an answer back to the exact place in your documents that supports it. They serve four functions at once: **trust** (users can verify), **debugging** (wrong citations expose retrieval or grounding failures immediately), **hallucination detection** (an answer that cannot cite is suspect), and **UX** (clicking a source builds confidence in the system). For personal-document RAG, citations are usually the feature that makes the difference between "a chatbot that is sometimes right" and "a research tool."

## 1. The approach landscape

### 1.1 Prompt-based inline markers (works with any LLM, local included)

The simplest mechanism: number the retrieved chunks, put them in the prompt with their ids, and instruct the model to append markers after claims:

```
System: Answer ONLY from the provided passages. After every sentence
containing factual claims, cite the supporting passage(s) like [1] or [2][4].
If the passages do not contain the answer, reply exactly:
"The documents don't contain this information."
— do not use outside knowledge.

Passage [1] (report_2023.pdf, p.14): ...
Passage [2] (minutes_2022.docx, p.3): ...
```

Strengths: zero extra machinery; works with Ollama/local models and every API. Weaknesses: the model sometimes drops markers, merges ids, or cites a passage that doesn't actually support the claim ("citation hallucination"); marker discipline degrades on weaker models; you should still verify mechanically (§1.4). Mitigations: keep the instruction block short and imperative, use few chunks (5–10), and post-validate that every cited id exists.

### 1.2 Structured citations — Claude API's native Citations feature

Claude's API can return citations as **structured objects with exact source spans**, guaranteed to point into documents you provided — the API parses the model's citation output and extracts `cited_text` directly, so citations are guaranteed to contain valid pointers, and `cited_text` does not count toward your output tokens (it's also not counted as input tokens when passed back in multi-turn).

How it works (from the [official docs](https://platform.claude.com/docs/en/build-with-claude/citations)):

1. Enable per document: add `"citations": {"enabled": true}` to a `document` content block (supported by all current Claude models — e.g., `claude-sonnet-5-5`, `claude-opus-5-5`, `claude-haiku-5-5`).
2. Feed retrieved chunks as document blocks — three types:
   - **Plain text**: the API auto-chunks into sentences; citations come back with `char_location` (0-indexed `start_char_index`, exclusive `end_char_index`).
   - **PDF** (base64/URL/`file_id`): text extracted, sentence-chunked; citations carry `page_location` (1-indexed start page, exclusive end). Note: Claude cannot cite images — scanned/image-only PDFs produce no citable text.
   - **Custom content**: you supply the block list, e.g. *your retrieved chunks* — citations then return `content_block_location` (`start_block_index`/`end_block_index`), which maps 1:1 back to your chunk store.
3. Use the optional `context` field to pass document-level context that the model may *use* but not *cite*.
4. Each response citation object contains: `type`, `cited_text`, `document_index`, `document_title`, and the location fields — everything a UI needs to render a deep link (jump to page N / highlight chars i–j).

Compared with the prompt approach, this removes id-drift and fake-pointer failure modes and simplifies the prompt (no citation rules needed). Constraint: it's a Claude API feature — your local Ollama generator cannot reproduce it; for fully-local stacks use §1.1 plus §1.4.

### 1.3 Quote-extraction pattern

A two-step prompt design that improves grounding with *any* model: first ask the model to extract **verbatim quotes** from the retrieved chunks that bear on the question (no answer yet); then ask it to compose the answer **using only** the extracted quotes, citing each. Forcing quotes-first raises the ceiling on faithfulness because the model commits to evidence before prose. Cost: a second LLM call (or one call with two output sections).

### 1.4 Post-hoc verification (attribution checking)

Treat the generated answer as claims to be audited: split the answer into sentences, and for each sentence test whether its cited chunk actually entails it, using an **NLI/entailment model** (or an LLM judge as a cheap substitute). Outcomes per sentence: *supported*, *unsupported* (flag, gray out, or strip), *no citation* (policy decision). This turns citations from decoration into a measurable accuracy number — the **citation accuracy** metric in [07 §3](07-evaluation.md) — and is the backbone of "verifiable RAG." Even a light version (check cited chunk text contains significant overlap with the claim's keywords/entities) catches the majority of fake citations.

## 2. Mapping citations back to your documents

Citations are only useful if the user can *go look*. That requires the metadata you stored at indexing time ([04 §3](04-indexing.md)): `doc_id → source_uri` (click to open the file), `page` (PDF viewers accept `#page=N`), and `char_start/char_end` (highlight the span in a viewer that supports it). Design targets:

| Citation returns | UI renders |
|---|---|
| `document_title` + chunk id | Superscript marker → tooltip with the chunk text |
| page number | Deep link `file.pdf#page=14` |
| char span (native Claude citations) | Highlighted span in an embedded viewer |
| none (model didn't cite) | Visual warning; the sentence is "unattributed" |

If your pipeline uses small-to-big retrieval ([05 §6](05-search-retrieval.md)), decide what a citation points at: the leaf chunk (precise) or the parent section (readable). A good compromise: cite the leaf, display the parent.

## 3. Refusal behavior: the other half of grounding

A grounded system must also *decline*. The prompt pattern in §1.1 includes an explicit refusal sentence; test it deliberately — your golden dataset should contain questions **whose answers are not in the corpus** ("not-answerable" set), and refusal correctness (answered-from-nothing vs. correctly declined) is a tracked metric, not an anecdote ([07 §4](07-evaluation.md)). Without this, systems fail in the worst way: confidently answering from parametric knowledge while your citations point at irrelevant chunks.

## 4. Language notes

Citations work identically for Turkish/mixed corpora — the mechanisms above are language-agnostic. Two practical details: (1) when models quote Turkish text verbatim, agglutinative suffixes mean "quotes" may be reformulated; if you need exact-quote fidelity (legal contexts), prefer mechanisms that return `cited_text` spans from the *source* (Claude native citations) over model-retyped quotes. (2) If your documents mix Turkish and English, keep chunk metadata's `language` field so you can diagnose cross-lingual retrieval misses separately ([03](03-embedding.md)).

## 5. Choosing an approach

| Situation | Approach |
|---|---|
| Claude API generator | Native Citations (§1.2) — structured, guaranteed pointers, cheaper than quote-prompting |
| Fully local generator (Ollama) | Prompt markers (§1.1) + post-hoc overlap/NLI verification (§1.4) |
| Legal/compliance-grade answers | §1.2 or §1.3 + full §1.4 audit + human review path |
| Quick prototype | §1.1, upgrade later — but keep `chunk_id/page/char` metadata in place from day one ([04 §3](04-indexing.md)), or citations can't be built retroactively |

## Sources

- [Claude Docs — Citations](https://platform.claude.com/docs/en/build-with-claude/citations)
- [Claude Docs — PDF support](https://platform.claude.com/docs/en/build-with-claude/pdf-support)
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval)
- RARR (attribution via search and revision) and AIS (attributable-to-identified-sources) literature for post-hoc attribution methods
