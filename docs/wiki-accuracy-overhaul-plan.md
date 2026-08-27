# Wiki Accuracy Overhaul — Implementation Plan

> **Status:** Draft (2026-06-08). Not yet implemented.
> **Scope:** Server-side wiki generation pipeline (embedding → retrieval → prompt → LLM → validation).
> **Owner:** TBD.
> **Goal:** Materially lift factual accuracy, structural stability, and depth of generated wikis without changing the embedding model or the broader architecture.

---

## 0. TL;DR

A read-only audit of the wiki pipeline (embedder, chunker, retriever, structure prompt, page prompt, model call) surfaced ~30 quality blockers. The single biggest one: we deploy a **reasoning model** (`gpt-5.4`) but never set `reasoning_effort`, so it defaults to `none` — we're paying for a reasoning SKU and using it as a vanilla chat model. Combined with junk in the index, header-overflowing chunks, no retrieval reranking, unguarded prompts, and zero post-generation validation, that's enough to explain the symptom.

The plan is **8 phases**, each independently shippable, sequenced so earlier phases create the conditions for later ones to land cleanly. The first PR (Phase 1 + Phase 2) is expected to deliver the bulk of the perceptible quality lift.

---

## 1. Background

### 1.1 Pipeline (current)

```
Repo clone
  └─► [Embedder] read_all_documents → split_code_at_boundaries → SafeEmbedder → FAISS / Azure AI Search
  └─► [Codemap]   ast_grep → codemap.json (used by structure prompt)
  └─► [WikiGenerator]
        ├─► build_file_tree (max_depth=6, max_entries=10000)
        ├─► read_readme (15K truncation)
        ├─► _call_llm( WIKI_STRUCTURE_PROMPT )                ← 1 call
        ├─► _parse_structure_xml (single best-effort retry on truncation)
        └─► for each page:
              ├─► RAG.call_with_file_filter (top_k_wiki=40 + file-filter chunks ≤80)
              ├─► format_context_text (no length cap)
              ├─► build_wiki_page_prompt (only page_title used)
              ├─► _call_llm( WIKI_PAGE_CONTENT_PROMPT )       ← 1 call/page
              └─► (optional review pass — disabled by default)
```

### 1.2 Deployed models

From [backend/config/infra.json](../backend/config/infra.json):

| Task | Deployment | Family | Temperature |
|---|---|---|---|
| Wiki structure + page gen | `gpt-5.4` | Reasoning | 1.0 (fixed) |
| Chat | `gpt-5.1-chat` | Reasoning | 1.0 (fixed) |
| Embedding | `text-embedding-3-large` (3072 dim) | — | n/a |

### 1.3 ⚠️ Model parameter constraints (verified 2026-06-08)

Per Microsoft docs ([Azure OpenAI reasoning models — Not Supported](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning?tabs=csharp%2Cgpt-5#not-supported)), **all GPT-5 series and o-series reasoning models reject** these parameters:

- `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `logprobs`, `top_logprobs`, `logit_bias`, `max_tokens`

The supported levers are:

| Parameter | Allowed values | Default on gpt-5.1+ | Notes |
|---|---|---|---|
| `reasoning_effort` | `none`, `low`, `medium`, `high` (`xhigh` on `gpt-5.1-codex-max`) | **`none`** | Biggest accuracy lever, currently unused |
| `verbosity` | `low`, `medium`, `high` | model-default | New GPT-5 knob for output length |
| `max_completion_tokens` | int | n/a | Use this, NOT `max_tokens` |
| `reasoning_summary` | `auto`, `concise`, `detailed` (Responses API) | none | Useful for debugging |

Both `gpt-5.4` and `gpt-5.1-chat` are on the reasoning-models availability list → same restriction. The current code at [`_call_llm` in backend/processor/wiki_generator.py:115](../backend/processor/wiki_generator.py#L115) only works because `temperature` defaults to `1.0`.

**Implication:** wiki output will **not** be byte-deterministic across runs (temperature is fixed at 1.0). The realistic determinism goal is *structurally stable* — same section count, same page IDs, same key cross-links — enforced via post-gen validators (Phase 6), not sampling.

### 1.4 `/no_think` is a Qwen-ism

The wiki LLM call hard-prepends `/no_think` to every prompt ([backend/processor/wiki_generator.py:124](../backend/processor/wiki_generator.py#L124)). That's a [Qwen-3](https://qwenlm.github.io/blog/qwen3/) thinking-toggle convention; OpenAI models treat it as literal prompt text. On GPT-5 it does NOT disable reasoning (that's controlled by `reasoning_effort`) but it does add noise to the user message. Same issue at [backend/modules/codetrace/service.py:90](../backend/modules/codetrace/service.py#L90).

### 1.5 Known bug catalog (found during audit)

These are referenced from the phase steps below and should be tracked in the corresponding PRs.

| # | Bug | File | Severity |
|---|---|---|---|
| B1 | `reasoning_effort` never set → reasoning model runs as chat model | [wiki_generator.py](../backend/processor/wiki_generator.py) `_call_llm` | **High** |
| B2 | `/no_think` prefix is Qwen-only — wasted tokens on GPT-5 | [wiki_generator.py](../backend/processor/wiki_generator.py#L124), [codetrace/service.py](../backend/modules/codetrace/service.py#L100) | Low |
| B3 | `codetrace/service.py` hardcodes `temperature: 0.7` — will fail on any reasoning deployment that actually validates the param | [codetrace/service.py:104](../backend/modules/codetrace/service.py#L104) | **High** (latent) |
| B4 | `excluded_dirs` only covers 10 dotfile dirs — `node_modules`, `__pycache__`, `dist`, `build`, etc. all reach the embedder | [excluded.json](../backend/config/excluded.json) | **High** |
| B5 | `excluded_files` adds `*.sql` but `included.json` re-adds `.sql` extension → huge SQL dumps embedded | [excluded.json](../backend/config/excluded.json), [included.json](../backend/config/included.json) | Medium |
| B6 | Enrichment header (50–200 tokens) NOT subtracted from chunker `target_tokens=2000` budget → real text ~1800 tok | [code_splitter.py](../backend/modules/embedder/code_splitter.py) | Medium |
| B7 | Method chunks lose parent class name (only chunk-local Functions list in header) | [code_splitter.py](../backend/modules/embedder/code_splitter.py) | Medium |
| B8 | OData filter built via f-string with no single-quote escape → injection / wrong matches on paths with `'` | [retriever.py](../backend/modules/embedder/retriever.py) `_call_with_file_filter_cloud` | **High** (security + correctness) |
| B9 | `call_with_file_filter` streams 80+ chunks to page LLM with no rerank → top-N dilution | [retriever.py](../backend/modules/embedder/retriever.py) | Medium |
| B10 | `format_context_text` has no length cap (comment says "no truncation") → blows past attention window on big repos | [chat/service.py](../backend/modules/chat/service.py) | Medium |
| B11 | `build_wiki_page_prompt` only reads `page_title`, ignores `description`/`importance`/`section_title`/`relatedPages` produced by structure-LLM | [wiki_page.py](../backend/promptstore/wiki_page.py) `build_wiki_page_prompt` | **High** |
| B12 | `enable_review_pass=False` default — review prompt exists but never runs | [wiki_generator.py](../backend/processor/wiki_generator.py#L350) | Medium |
| B13 | `_parse_structure_xml` accepts pages with empty `<file_path>` lists — they reach RAG with `filePaths=[]` and rely on semantic-only retrieval | [wiki_generator.py](../backend/processor/wiki_generator.py) | Medium |
| B14 | `build_file_tree` hard `max_depth=6` drops deep modules entirely for tall repos | [wiki_generator.py:54](../backend/processor/wiki_generator.py#L54) | Medium |
| B15 | `read_readme` truncates at 15K chars (head only) — drops API tables, examples, FAQ at bottom | [wiki_generator.py:97](../backend/processor/wiki_generator.py#L97) | Low |
| B16 | No citation validator — fabricated `[file.ext L#-L#]()` ships verbatim | new | Medium |
| B17 | No Mermaid validator — broken diagrams ship | new | Medium |
| B18 | Each page generated in isolation → repeated content across Overview / System Architecture / Core Features | [wiki_generator.py](../backend/processor/wiki_generator.py) per-page loop | Medium |

---

## 2. The 8-phase plan

Each phase has: **Why**, **Steps** (with file + symbol references), **Verification** (objective checks), **Risk**, and **Rough effort**.

---

### Phase 1 — Model call hygiene + reasoning activation

**Bugs addressed:** B1, B2, B11, B12. Touches the highest-leverage code path.

**Why.** A reasoning model running with `reasoning_effort=none` is the most expensive way to get chat-model output. Activating it is a one-line change with the largest expected impact on factual depth. Wiring the page prompt to consume the structure LLM's `description` / `importance` / `section_title` corrects a long-standing context-loss bug.

**Steps.**

1. **Verify the hypothesis first.** Make one explicit call to `gpt-5.4` with `reasoning_effort='medium'` and `reasoning_summary='auto'`, inspect `output_tokens_details.reasoning_tokens` in the response. Confirm:
   - `reasoning_tokens == 0` on default call (proves current state is "no reasoning")
   - `reasoning_tokens > 0` when effort is explicitly set (proves the lever works)
   - This justifies the rest of the phase.

2. **Add `reasoning_effort` + `verbosity` support to `_call_llm`** in [backend/processor/wiki_generator.py](../backend/processor/wiki_generator.py#L114). Current signature:

   ```python
   def _call_llm(prompt: str, model_client: AzureAIClient,
                 deployment: str, temperature: float = 1.0,
                 max_tokens: int = 16384) -> Tuple[str, str]:
   ```

   Target signature:

   ```python
   def _call_llm(
       prompt: str, model_client: AzureAIClient, deployment: str,
       *,
       reasoning_effort: Optional[Literal['none','low','medium','high']] = None,
       verbosity: Optional[Literal['low','medium','high']] = None,
       max_completion_tokens: int = 16384,
       temperature: Optional[float] = None,  # only used for non-reasoning models
   ) -> Tuple[str, str]:
   ```

3. **Add `is_reasoning_model(deployment_name)` helper** in [backend/config.py](../backend/config.py) (or new `backend/utils/model_capabilities.py`). Match on deployment name prefix (`gpt-5`, `gpt-5.`, `o1`, `o3`, `o4`, …). Inside `_call_llm`:
   - if reasoning model → emit `reasoning_effort` + `verbosity` + `max_completion_tokens`, **never** `temperature`
   - if non-reasoning model → emit `temperature` + `max_completion_tokens`

4. **Wire task-specific defaults** in the wiki gen orchestration:
   - Wiki structure call ([wiki_generator.py:424 / 446](../backend/processor/wiki_generator.py#L424)) → `reasoning_effort='high'`, `verbosity='medium'`
   - Wiki page call ([wiki_generator.py:570](../backend/processor/wiki_generator.py#L570)) → `reasoning_effort='medium'`, `verbosity` driven by `page.importance` (`'high'` for `high`, `'medium'` for `medium`, `'low'` for `low`)
   - Wiki page review call ([wiki_generator.py:586](../backend/processor/wiki_generator.py#L586)) → `reasoning_effort='low'`, `verbosity='medium'`

5. **Drop the `/no_think` prefix.** Remove the literal in [wiki_generator.py:124](../backend/processor/wiki_generator.py#L124) and [codetrace/service.py:100](../backend/modules/codetrace/service.py#L100). Same for any other prompt site (grep `/no_think`).

6. **Fix codetrace temperature bug (B3).** [codetrace/service.py:104](../backend/modules/codetrace/service.py#L104) hardcodes `'temperature': 0.7`. Replace with the same `is_reasoning_model` branching as Phase 1 step 3.

7. **Flip `enable_review_pass=True`** default in [wiki_generator.py:350](../backend/processor/wiki_generator.py#L350). The review prompt is already implemented at `WIKI_PAGE_REVIEW_PROMPT` in [backend/promptstore/wiki_page.py](../backend/promptstore/wiki_page.py); the call site at [wiki_generator.py:586](../backend/processor/wiki_generator.py#L586) is in place.

8. **Wire `page_description`, `page_importance`, `section_title` into the page prompt** ([backend/promptstore/wiki_page.py](../backend/promptstore/wiki_page.py) `build_wiki_page_prompt`, currently around line 194). Update signature:

   ```python
   def build_wiki_page_prompt(
       page_title: str,
       page_description: str,
       page_importance: Literal['high','medium','low'],
       section_title: str,
       related_page_titles: List[str],   # NEW — small list, not full catalog
       file_paths_list: str,
       language_name: str,
       page_catalog: str,                # keep full catalog as separate block
       ...
   ) -> str:
   ```

   Update the prompt body to include:
   ```text
   PAGE INTENT
   This page is the {page_importance}-importance entry under section
   "{section_title}". Its purpose is: {page_description}

   STRONGLY RELATED PAGES (prefer linking these in cross-references)
   - {related_page_title_1}
   - {related_page_title_2}
   ...
   ```

9. Update the caller at [wiki_generator.py:556](../backend/processor/wiki_generator.py#L556) to pass the new fields. The structure-parse already extracts them — they just aren't propagated.

**Verification.**

- API-level: confirm `output_tokens_details.reasoning_tokens > 0` in the LLM response (log it).
- Run wiki gen twice on the same repo. Diff: structure should be structurally similar (same section count ±1, same set of page IDs within 90%). Page content will not be byte-equal.
- Manual rubric on 5 pages: factual depth, presence of intent-aligned content (matches `description`), use of related-page cross-links.
- Cost diff: expect 2–5× more output tokens per page (reasoning tokens are billed). Document the new $/wiki figure.

**Risk.** Low. All changes are additive parameters or removing dead code. Worst case: rollback `reasoning_effort` if cost is unacceptable.

**Rough effort.** Small. ~1 day for a confident engineer including verification.

**Files touched.**
- [backend/processor/wiki_generator.py](../backend/processor/wiki_generator.py) — `_call_llm`, `generate_wiki`, per-page loop near line 556–586, default `enable_review_pass=True`
- [backend/promptstore/wiki_page.py](../backend/promptstore/wiki_page.py) — `build_wiki_page_prompt` signature + body
- [backend/modules/codetrace/service.py](../backend/modules/codetrace/service.py) — drop `/no_think`, fix `temperature: 0.7`
- new: `backend/utils/model_capabilities.py` (or extend `backend/config.py`)
- [backend/config/infra.json](../backend/config/infra.json) — optional: add `reasoning_effort` defaults per task

---

### Phase 2 — Embedder noise filtering

**Bugs addressed:** B4, B5, plus a perf win.

**Why.** Index quality has a ceiling set by what you embed. Today [`excluded_dirs` in backend/config/excluded.json](../backend/config/excluded.json) only excludes 10 dotfile dirs. Real-world repos pull `node_modules`, `__pycache__`, `dist`, `build`, `target`, `bin`, `obj`, `.next`, `vendor`, `Pods`, `coverage`, `htmlcov`, `.pytest_cache`, `.mypy_cache`, `migrations` into the index, dilute retrieval, and bloat memory. The `.sql` extension is excluded by file pattern but re-added by `included.json`'s code-extension list (B5).

**Steps.**

1. **Extend `excluded_dirs`** in [backend/config/excluded.json](../backend/config/excluded.json) with the standard build/cache/vendor set:
   ```
   node_modules, __pycache__, .pytest_cache, .mypy_cache, .tox,
   dist, build, target, bin, obj, out,
   .next, .nuxt, .svelte-kit, .turbo, .vercel,
   vendor, Pods, Carthage,
   coverage, htmlcov, .nyc_output,
   migrations, Migrations,
   .gradle, .idea, .vscode,
   __snapshots__, __mocks__, __fixtures__
   ```
   Document the rationale in a comment block at the top of the file.

2. **Resolve `.sql` conflict (B5).** Either remove `.sql` from `included.json`'s code-extension list OR remove `*.sql` from `excluded.json`'s `excluded_files`. Recommendation: **keep `.sql` excluded** for wiki gen (migration dumps dominate by line count and rarely add architectural insight); revisit if a customer wiki specifically needs schema documentation.

3. **Add a per-file SIZE CAP to `read_all_documents`** in [backend/modules/embedder/document.py](../backend/modules/embedder/document.py). Skip files larger than:
   - Code files: 1 MB
   - Doc files (`.md`, `.rst`, `.txt`): 200 KB
   - JSON / YAML: 500 KB

   Log skipped files with size + reason. This kills the giant-SQL-hang scenario captured in `/memories/repo/embedding_stuck_huge_sql.md` and drops most generated-asset noise.

4. **Improve test detection.** Mark a file `is_implementation=False` when path matches any of:
   ```
   ^tests?/, /tests?/, _test\., \.test\., \.spec\.,
   conftest\.py, __mocks__/, __fixtures__/, /e2e/
   ```
   Wiki-gen mode: exclude from index entirely (default).
   Chat mode: keep but tag for deprioritization. Implement as a config flag `embedder.exclude_tests_for_wiki: true`.

5. **Cache the tiktoken encoder** at module level in [backend/modules/embedder/tokenizer.py](../backend/modules/embedder/tokenizer.py):
   ```python
   _ENCODING = tiktoken.get_encoding("cl100k_base")
   def count_tokens(text: str) -> int:
       return len(_ENCODING.encode(text))
   ```
   Cheap perf win + removes the stale "model-specific encoding" comment.

**Verification.**

- Re-embed a known repo before/after. Expect 20–40% chunk count drop on a real-world repo with tests + generated code.
- Sample 20 retrieved chunks for a known wiki page; confirm 0 chunks from `node_modules`, `dist`, `coverage`, `migrations`.
- Log diff: confirm the size-cap log lines fire for the expected files.

**Risk.** Low. All exclusions are conservative.

**Rough effort.** Small. ~0.5 day.

**Files touched.**
- [backend/config/excluded.json](../backend/config/excluded.json)
- [backend/config/included.json](../backend/config/included.json) (resolve `.sql` conflict)
- [backend/modules/embedder/document.py](../backend/modules/embedder/document.py) — `read_all_documents` filter loop
- [backend/modules/embedder/tokenizer.py](../backend/modules/embedder/tokenizer.py) — `count_tokens`

---

### Phase 3 — Chunk content quality

**Bugs addressed:** B6, B7, plus doc-chunk lineage loss.
**Depends on:** Phase 2 (run on a cleaner corpus).

**Why.** Even with the right files, today's chunks bleed semantic context. A method body chunk often loses its parent class name (only the chunk-local `Functions:` list is in the header). The 50–200 token enrichment header is NOT subtracted from the splitter's 2000-token target, so real text is ~1800 tokens but the splitter believes it's spending 2000. Markdown doc chunks lose H1→H2 lineage so retrieval can't tell apart two "Configuration" sections.

**Steps.**

1. **Reserve header budget in `split_code_at_boundaries`** ([backend/modules/embedder/code_splitter.py](../backend/modules/embedder/code_splitter.py)). Add an `enrichment_overhead: int = 150` parameter; subtract from effective `target_tokens` and `max_tokens` before doing boundary search.

2. **Track ancestor class stack** while walking AST sections. When a section is inside a class, prepend to the section data:
   ```
   Parent class: {ClassName}
   {first line of class docstring, if any}
   ```
   This restores semantic context for method-only chunks and improves both embedding and prompt-side comprehension.

3. **Heading-stack tracking for docs.** In `_split_doc_text` (or equivalent), track the H1→H2→H3 stack. Each chunk header gets:
   ```
   Heading: # Page Title / ## Section / ### Subsection
   ```
   Lets retrieval disambiguate "Configuration" under "Auth" vs under "Caching".

4. **Fix SafeEmbedder duplicate-vector handling** in [backend/clients/embedding_client.py](../backend/clients/embedding_client.py) `_prepare_safe_inputs`. When `split_into_chunks` produces N sub-chunks of one logical chunk:
   - Stamp each sub-chunk with the SAME `file_path` + `chunk_index` but distinct `sub_chunk_index`
   - Update the retriever dedup key in [retriever.py](../backend/modules/embedder/retriever.py) `_doc_key` to `(file_path, chunk_index, sub_chunk_index)` so both pieces survive and ordering is preserved
   - Alternative: bump per-chunk size policy so the 7500-tok split path is rare. Today it mainly affects huge YAML/JSON; defer the alternative unless metrics show it matters.

5. **Document the chunking invariants** in [backend/modules/embedder/README.md](../backend/modules/embedder/README.md) so future edits don't drift.

**Verification.**

- Random-sample 20 chunks across languages; assert every method chunk's header names the enclosing class.
- Re-embed and check 95th percentile chunk token count is ≤ 2050.
- Spot-check 5 doc-chunk retrievals; confirm heading breadcrumb is present.

**Risk.** Medium. Touches the most-exercised code path. Requires regression test on at least 3 representative repos.

**Rough effort.** Medium. ~2 days including regression tests.

**Files touched.**
- [backend/modules/embedder/code_splitter.py](../backend/modules/embedder/code_splitter.py) — `split_code_at_boundaries`, `build_enriched_chunk_text`, `_split_doc_text`
- [backend/clients/embedding_client.py](../backend/clients/embedding_client.py) — `SafeEmbedder._prepare_safe_inputs`, `split_into_chunks`
- [backend/modules/embedder/retriever.py](../backend/modules/embedder/retriever.py) — `_doc_key` for dedup
- [backend/modules/embedder/README.md](../backend/modules/embedder/README.md) — invariants doc

---

### Phase 4 — Retrieval quality

**Bugs addressed:** B8 (security), B9, B10, plus parity gaps.

**Why.** With `top_k_wiki=40` for BOTH the file-filtered query AND the unfiltered semantic query, `call_with_file_filter` can stream 80 chunks (post-dedup ~50–70) into the page LLM with NO rerank. The top 10 are usually great; the rest are noise that dilutes attention. Cloud mode also has the unescaped-OData bug.

**Steps.**

1. **OData injection/escape fix (B8)** in [backend/modules/embedder/retriever.py](../backend/modules/embedder/retriever.py) `_call_with_file_filter_cloud`. Today:
   ```python
   filter_expr = f"filepath eq '{fp}'"   # broken: no escape for `'` in fp
   ```
   Fix:
   ```python
   def _odata_escape_string(s: str) -> str:
       return s.replace("'", "''")
   filter_expr = f"filepath eq '{_odata_escape_string(fp)}'"
   ```
   Add the same helper to any other filter builder in [backend/clients/search_client.py](../backend/clients/search_client.py). Reject paths containing control chars (`\x00–\x1f`) defensively.

2. **Normalize `file_path` on both sides of the index.** Define a single `_norm(path)` (forward slashes, no leading `./`, no double slashes; decide casefolding policy explicitly — case-sensitive on POSIX repos, case-fold for Windows-source repos via a config flag). Apply at:
   - chunk creation time (before storing)
   - lookup time in `_file_path_index`
   - cloud filter builder

3. **Cap and rerank merged retrieval** in [retriever.py](../backend/modules/embedder/retriever.py) `call_with_file_filter`. After merging file-filtered + semantic results, run a cheap rerank and keep top-25 instead of 50+:
   - **Phase A (default):** BM25 over `(page_title + page_description + file_path_basename + first_120_chars_of_chunk)`. Deterministic, no LLM call, ~10ms.
   - **Phase B (optional, gated):** cross-encoder rerank via Azure OpenAI (1 extra LLM call per page; only worth it if BM25 isn't enough).
   - **Phase C (optional, gated):** local sentence-transformer reranker (adds dep).

   Start with A; measure; only adopt B/C if metrics show retrieval is the bottleneck after A.

4. **Smarter dedup in `format_context_text`** ([backend/modules/chat/service.py](../backend/modules/chat/service.py)). Detect overlapping `(file_path, start_line, end_line)` ranges; merge into a single "Span: lines A–B" entry rather than emitting both.

5. **`top_k` parity.** Make local FAISS use the SAME effective `top_k` as cloud for wiki gen. Today [embedder.json](../backend/config/embedder.json) says 12 but wiki passes 40 — confirm the override path. Document the per-task top_k policy in `embedder.json`:
   ```json
   {
     "top_k": 12,            // chat
     "top_k_wiki": 40,       // wiki page gen (pre-rerank)
     "top_k_wiki_final": 25  // post-rerank cap (Phase 4 step 3)
   }
   ```

6. **Enable Azure AI Search Semantic Ranker (L2)** in [backend/clients/search_client.py](../backend/clients/search_client.py) `search_as_documents`. Requires Standard tier. Gate with a config flag `azure_ai_search.semantic_ranker: true`. Big quality win on hybrid queries when available.

7. **Per-page max context cap** in `format_context_text`. Soft cap at 60K chars; if over, prefer keeping file-filtered (declared) chunks over semantic-only ones. Today the comment says "no truncation — quality is critical" but past ~32K most models' attention degrades — measure before lifting the cap.

**Verification.**

- B8 regression test: file path `it's-mine/file.py` should round-trip through the filter without error.
- Pick 5 wiki pages; for each, dump retrieved chunks before/after rerank and dedup. Expect 30–50% fewer chunks, same or better topical match (manual judgment).
- End-to-end wiki gen: page generation logs `context=` size should be steadier across pages (less variance).
- Cloud parity: run identical flow under `azure_ai_search.enabled=true`; confirm retrieved chunk sets agree with local within tolerance.

**Risk.** Medium (B8 fix is straightforward but other steps require A/B measurement to be sure rerank improves rather than degrades quality).

**Rough effort.** Medium. ~2–3 days. Step 6 (semantic ranker) is small if SKU upgrade is already done.

**Files touched.**
- [backend/modules/embedder/retriever.py](../backend/modules/embedder/retriever.py) — `call_with_file_filter`, `_call_with_file_filter_cloud`, `_file_path_index`, `_doc_key`
- [backend/clients/search_client.py](../backend/clients/search_client.py) — index schema, `search_as_documents`, filter builders
- [backend/modules/chat/service.py](../backend/modules/chat/service.py) — `format_context_text`
- [backend/config/embedder.json](../backend/config/embedder.json) — `top_k`, `top_k_wiki`, `top_k_wiki_final`

---

### Phase 5 — Prompt engineering

**Bugs addressed:** Hallucination guard, no-example prompts, lopsided Mermaid block, weak page catalog format.
**Parallel with Phase 4.**

**Why.** Today's prompts are dense rule lists with no examples and no explicit hallucination guard. The Mermaid block alone is ~50 lines (arrow types, activation boxes, structural elements) and buries everything else. The LLM has zero context about WHY a given page exists (Phase 1 fixes half by passing description; this phase tightens the rest).

**Steps.**

1. **Add an explicit hallucination guard** at the top of `WIKI_PAGE_CONTENT_PROMPT` in [backend/promptstore/wiki_page.py](../backend/promptstore/wiki_page.py):
   ```
   GROUNDING RULES (override all other instructions)
   - If a fact is not directly supported by SOURCE CODE CONTEXT or the
     declared file list, do NOT include it.
   - Prefer "Not specified in the provided sources" over guesses.
   - Every architectural claim must cite a Sources: line pointing to a
     file from the declared file list.
   ```

2. **Add 1 few-shot example** to:
   - `WIKI_STRUCTURE_PROMPT` in [backend/promptstore/wiki_structure.py](../backend/promptstore/wiki_structure.py) — a small valid `<wiki_structure>` (compact ~30 lines).
   - `WIKI_PAGE_CONTENT_PROMPT` — a small Markdown page including the `<details>` block, one `## Section`, one Mermaid diagram, one `Sources:` line.

   Examples beat rules for LLM behavior steering.

3. **Tighten the Mermaid block.** Collapse the 50-line rule list into a short bullet section + one concrete `sequenceDiagram` example. Keep the syntax rules that catch the actual bugs we see (no `graph LR` without nodes declared, no `loop`/`end` mismatch). Drop the encyclopedic enumeration.

4. **Per-page codemap injection.** When generating page X, compute a small `page_codemap_summary` from the page's declared files (their key classes/functions + 1-hop imports) and inject under a new "PAGE CODE MAP" block. Reuse the `edge_index` already built in `generate_wiki`. Today codemap is only in the structure prompt; page prompt doesn't see it.

   Add helper `summarize_codemap_for_files(file_paths, edge_index, codemap)` in [backend/processor/codemap_generator.py](../backend/processor/codemap_generator.py).

5. **Per-section length hints** in the page prompt:
   ```
   FORMAT
   - Each H2 section: 150–500 words.
   - Total page: 800–2500 words depending on importance (this page = {importance}).
   ```

6. **Strengthen `page_catalog` format.** Today `format_page_catalog()` emits `- id: title`. Make it:
   ```
   - [{importance}] {id}: {title} — {short_description}
   ```
   The LLM picks meaningful cross-link targets instead of catalog noise.

7. **Output structural anchor for truncation detection.** Require the LLM to end every page with:
   ```html
   <!-- generated:complete -->
   ```
   Phase 6 validators check it. Absence triggers a single retry with `max_completion_tokens` bumped.

**Verification.**

- A/B compare wiki output for 1 benchmark repo with old vs new prompts. Score:
  - factual accuracy (manual spot check on 5 pages, 1–5 scale)
  - Mermaid syntax error count (target: 0)
  - cross-link density per page (target: ≥2 meaningful in-page links)
  - presence of fabricated file paths (target: 0)
- Read 3 pages end-to-end; confirm sections fall in the 150–500 word band.

**Risk.** Low–Medium. Prompt changes can regress unexpectedly; A/B comparison on a benchmark repo is mandatory.

**Rough effort.** Small–Medium. ~1.5 days.

**Files touched.**
- [backend/promptstore/wiki_page.py](../backend/promptstore/wiki_page.py) — `WIKI_PAGE_CONTENT_PROMPT`, `build_wiki_page_prompt`, `format_page_catalog`
- [backend/promptstore/wiki_structure.py](../backend/promptstore/wiki_structure.py) — `WIKI_STRUCTURE_PROMPT`, `WIKI_STRUCTURE_CONCISE_PROMPT`
- [backend/processor/wiki_generator.py](../backend/processor/wiki_generator.py) — per-page loop, compute per-page codemap snippet
- [backend/processor/codemap_generator.py](../backend/processor/codemap_generator.py) — add `summarize_codemap_for_files`

---

### Phase 6 — Post-generation validation

**Bugs addressed:** B13, B16, B17, plus truncation detection.

**Why.** Today's only check is "structure_xml length > 200 chars; retry once with dir-only tree" ([wiki_generator.py](../backend/processor/wiki_generator.py) around line 446). Generated pages are stored verbatim — fabricated citations, broken Mermaid, truncated content all ship to the user.

**Steps.**

1. **Citation validator.** New module `backend/modules/wiki/validators.py`:
   ```python
   def validate_citations(page_markdown: str, allowed_file_paths: set[str]) -> tuple[str, list[str]]:
       """Returns (cleaned_markdown, list_of_invalid_citations)."""
   ```
   Parse all `Sources: [file.ext L#-L#](url)` and `[name](deepwiki://id)` links. Reject citations whose `file.ext` is not in the page's `file_paths` (declared or expanded). Replace bad ones with `Sources: [file.ext]()` or strip them.

2. **Mermaid validator.**
   ```python
   def validate_mermaid_blocks(page_markdown: str) -> tuple[str, list[str]]:
       """Returns (cleaned_markdown, list_of_broken_blocks)."""
   ```
   Extract all ` ```mermaid ... ``` ` blocks. Cheap structural checks:
   - every `participant` declared before use
   - no bare `graph LR` without node declarations
   - balanced `loop`/`end`, `alt`/`end`, `opt`/`end`, `par`/`end`
   - no unicode quotes (LLM occasionally emits `"` instead of `"`)

   On failure: single retry with a small "fix this diagram" sub-prompt rather than regenerating the whole page.

3. **Markdown structural check.**
   - Ensure present `<details>` block, present `# {page_title}` H1
   - Strip stray ` ```markdown ` fences (page LLM ignores the no-fence instruction sometimes)
   - Auto-fix what's mechanical; reject what isn't and trigger retry.

4. **Structure XML validator.** In `_parse_structure_xml()` ([backend/processor/wiki_generator.py](../backend/processor/wiki_generator.py)), reject parsed result if:
   - a `<page>` has no `<file_path>` AND no `<related>`
   - any `<page_ref>` points to non-existent page id
   - any page_id duplicate after dash→dot normalization
   - structure has 0 sections or 0 pages

   On reject: log diagnostic + retry with explicit "previous attempt failed validation because X" message in the prompt. Augment [backend/modules/wiki/xml_repair.py](../backend/modules/wiki/xml_repair.py) with these rules.

5. **Truncation detection.** Check for the `<!-- generated:complete -->` marker (added in Phase 5). If missing AND content length is within 5% of `max_completion_tokens` ceiling, retry the page once with `max_completion_tokens=24576`.

**Verification.**

- Run validators in dry-run mode first on existing wiki output; count failures per category.
- Then enable enforcement; confirm shipped pages have 0 bad citations and 0 broken Mermaid blocks.
- Retry rate should be < 10% of pages; if higher, treat as a prompt-engineering bug (back to Phase 5).

**Risk.** Low. All net-new code with clear failure modes.

**Rough effort.** Medium. ~2 days.

**Files touched.**
- new: [backend/modules/wiki/validators.py](../backend/modules/wiki/validators.py)
- [backend/processor/wiki_generator.py](../backend/processor/wiki_generator.py) — call validators in per-page loop and after structure parse
- [backend/modules/wiki/xml_repair.py](../backend/modules/wiki/xml_repair.py) — augment with new structural rules
- [backend/promptstore/wiki_page.py](../backend/promptstore/wiki_page.py) — small "fix this diagram" / "fix these citations" sub-prompts

---

### Phase 7 — Structure-generation upgrades

**Bugs addressed:** B14, B15, plus zero-file-page handling.

**Why.** A bad structure ceiling-caps the whole wiki: missing pages, wrong file→page assignments, deep modules left out. Today `max_depth=6` in [build_file_tree](../backend/processor/wiki_generator.py#L54) and 15K-char README head-truncation at [read_readme](../backend/processor/wiki_generator.py#L97) discard real context.

**Steps.**

1. **Smarter file tree.** Replace fixed `max_depth=6` with an adaptive policy in `build_file_tree`:
   - descend until total entries > 8000
   - then start collapsing the deepest unimportant dirs (test/, doc/, example/, sample/)
   - always keep paths of files mentioned in codemap top-symbols
   - emit a "[... N more files hidden in collapsed dirs]" marker so the LLM knows it's not seeing everything

2. **Smarter README handling.** `read_readme` currently keeps `[:15000]`. Replace with: keep first 5K + last 3K + an LLM-extracted summary of the middle if oversize. The middle-summary is a single 500-token LLM call; cheap.

3. **Inject high-level codemap before structure prompt.** Already done — verify it actually fits in the prompt budget and isn't getting truncated by SafeEmbedder. Add a log of the structure-prompt total size in characters.

4. **Validate structure-LLM output against file tree.** Every `<file_path>` in `<page>` must exist in `repo_path`. Today this filter happens AFTER parse but pages with 0 valid files still go to RAG with `filePaths=[]`. Instead: when a page has 0 valid files, attempt fuzzy match (basename + nearest parent dir) before falling back to semantic-only. If still 0, log a structured warning and drop the page (or mark for human review).

5. **Detect duplicate / overlapping pages.** Post-parse, if two pages have ≥60% overlap in declared files, log a warning. Optional: merge them with the larger page absorbing the smaller. Default to log-only; let the user enable merging via config.

**Verification.**

- Re-run structure gen for a deep repo (≥7 directory levels). Confirm more pages reference files in the deep layers.
- Re-run for a repo with a >20K-char README. Confirm structure no longer omits late-README topics.
- Confirm fewer "Pages with no matching files" warnings in the log.

**Risk.** Low–Medium. Adaptive file tree is the riskiest step; needs a fallback that always produces a parseable tree.

**Rough effort.** Medium. ~1.5 days.

**Files touched.**
- [backend/processor/wiki_generator.py](../backend/processor/wiki_generator.py) — `build_file_tree`, `read_readme`, structure validation block, `_parse_structure_xml`
- [backend/promptstore/wiki_structure.py](../backend/promptstore/wiki_structure.py) — `file_tree_dirs_only`

---

### Phase 8 — Multi-page consistency

**Bugs addressed:** B18.

**Why.** Today each page is generated in isolation. Result: same architectural overview repeated on Overview, System Architecture, AND Core Features; inconsistent terminology ("user record" vs "user entity"); broken cross-links.

**Steps.**

1. **Two-pass generation.**
   - Pass 1: generate all pages as today.
   - Pass 2 (single LLM call): with the FULL set of page titles + intro paragraphs as context, produce:
     - a glossary
     - a per-page "delta hints" list ("Page 2.1 already covers JWT issuance, don't repeat it on 4.3")
   - Optional: only re-generate `high`-importance pages with the delta hints injected; cheaper pages keep pass-1 content.

2. **Shared terminology dictionary.** Extract top entity names (classes, services, modules) from the codemap before the per-page loop. Inject a small "TERMINOLOGY" block in every page prompt:
   ```
   TERMINOLOGY (use these exact names)
   - AuthService (not "authentication service" or "AuthSvc")
   - UserRepository
   - JobQueue
   ...
   ```
   Add helper `extract_top_terminology(codemap) -> dict[str, str]` in [backend/processor/codemap_generator.py](../backend/processor/codemap_generator.py).

3. **Cross-link audit pass.** After all pages generated:
   - parse all `[Title](deepwiki://id)` references
   - build per-page back-link counts
   - pages with 0 back-links are likely orphans — log and optionally regenerate the structure's `relatedPages` to surface them

4. **Final assembly check.** Verify:
   - every section's overview page exists
   - every `<page_ref>` resolves
   - every page is reachable from a root section

**Verification.**

- Pick 3 pages that historically overlap (Overview, System Architecture, Backend Systems). After Phase 8, manual diff — confirm no >2-sentence verbatim repeats; same entities use same names.
- Cross-link audit log: orphan page count should be ≤ 10% of total pages.

**Risk.** Medium (pass-2 regeneration could introduce regressions on already-good pages). Mitigate by gating on `importance == 'high'`.

**Rough effort.** Medium. ~2 days.

**Files touched.**
- [backend/processor/wiki_generator.py](../backend/processor/wiki_generator.py) — orchestration of the two passes
- new helper in [backend/promptstore/wiki_page.py](../backend/promptstore/wiki_page.py) for the consistency pass prompt
- [backend/processor/codemap_generator.py](../backend/processor/codemap_generator.py) — `extract_top_terminology`

---

## 3. Sequencing

**Independent / can ship now:** Phases 1, 2, 6 (validators), 7 (file tree + readme).

**Chain order recommended:** 1 → 5 (prompts) → 4 (retrieval) → 3 (chunks). Each lifts the next.

**Last:** Phase 8 (consistency). Most impactful AFTER everything else is clean; running it on noisy output amplifies noise.

**Practical PR breakdown:**

| PR | Contents | Why grouped |
|---|---|---|
| PR #1 | Phase 1 (model + reasoning_effort) + Phase 2 (excluded dirs / size cap / test detection) | Highest ROI, lowest risk, ships fastest |
| PR #2 | Phase 6 (validators) + Phase 7 (structure quality) | Both server-side, both observable in logs |
| PR #3 | Phase 5 (prompts) + Phase 4 (retrieval rerank/dedup/escape) | Both shape what the LLM sees |
| PR #4 | Phase 3 (chunk overhaul) | Requires re-embed of test repos; ship alone |
| PR #5 | Phase 8 (consistency) | Final polish on top of clean pipeline |

---

## 4. Verification (end-to-end)

1. **Pick a benchmark repo.** Recommend `msdata/orcasql` from existing logs (it's been the recurring stress test). Add 1 small repo (≤200 files) and 1 large repo (≥2000 files) for breadth.

2. **Baseline.** Regenerate wiki on `main` for each benchmark repo. Save JSON output to `test/baseline_<repo>_<date>.json`.

3. **After each PR.** Regenerate wiki, diff against baseline. Track:

   | Metric | Target | Notes |
   |---|---|---|
   | Pages with `filePaths=[]` | ≤ 10% of pages | from log |
   | Average chunks in retrieval per page | 15–25 (down from 40–80) | from log |
   | Mermaid syntax error count | 0 | from Phase 6 validator |
   | Fabricated citation count | 0 | from Phase 6 validator |
   | Structural determinism (2 runs) | ≤ 5% diff in page IDs / section structure | manual |
   | Manual rubric on 5 hand-picked pages | factual accuracy, completeness, cross-link relevance (1–5 each) | required for sign-off |
   | `reasoning_tokens` per page | > 0 | from API response (proves Phase 1 working) |
   | Cost per wiki generation | track | expect 2–5× post-Phase-1 |

4. **Cloud parity.** Run identical flow under `azure_ai_search.enabled=true`; confirm cloud results match local within tolerance (same metrics).

5. **Sign-off rubric.** A PR ships when:
   - all automated metrics meet target
   - manual rubric scores ≥ 3.5/5 average across 5 pages
   - no regression on the small-repo benchmark (catches over-fitting to the big one)

---

## 5. Decisions & assumptions

- Embedding model `text-embedding-3-large` @ 3072 dim is correct and matches schema (verified). **NOT changing the model in this plan.**
- Reasoning model is `gpt-5.4`; `gpt-5.1-chat` stays for chat endpoint. Page/structure gen will follow Phase 1's tuned settings.
- **Out of scope:** storage layer refactor, switching off Azure AI Search, multi-language wikis (single-language per run today).
- **Cost.** Enabling `enable_review_pass=True` + `reasoning_effort='medium'` + per-page codemap injection adds an expected 2–5× LLM token spend per page. Acceptable since wiki gen is offline / cached. Track actuals post-Phase-1 and revisit if > 5×.

---

## 6. Open questions (decide before implementation)

1. **Test files in the index** (Phase 2 step 4). Option A: hard-exclude in wiki-gen mode (proposed default — cleaner wiki). Option B: keep but downweight via metadata flag (better for chat). **Recommendation: A for now**, revisit if chat quality drops.

2. **Semantic Ranker on Azure AI Search** (Phase 4 step 6). Requires Standard SKU upgrade. **Confirm SKU & budget before implementing.**

3. **Reranker model for local FAISS** (Phase 4 step 3). Options:
   - (a) BM25 over title+filename only — cheap, deterministic
   - (b) cross-encoder via Azure OpenAI — 1 extra LLM call per page, slow
   - (c) local sentence-transformer model — adds dep

   **Recommendation: start with (a)**, evaluate need for (b) after measurement.

4. **Verify `gpt-5.4` `reasoning_effort` default** (Phase 1 step 1). The MS doc explicitly states `gpt-5.1` defaults to `'none'`; `gpt-5.4` is assumed to inherit. If verification shows otherwise (e.g., `'medium'` default), recalibrate the per-task settings in Phase 1 step 4.

5. **`max_completion_tokens` ceiling** (Phase 1 step 2 / Phase 6 step 5). 16K today, 24K on retry. Confirm the deployment's actual hard cap from Azure portal before relying on 24K.

6. **codetrace path uses the reasoning model** (B3). If `temperature: 0.7` was working until now, it's because the SDK / proxy silently drops it. Worth confirming with one explicit test call to ensure removal doesn't change behavior in an unexpected way.

---

## 7. References

- [Azure OpenAI reasoning models](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning?tabs=csharp%2Cgpt-5) — `reasoning_effort`, `verbosity`, supported/unsupported parameters
- [backend/config/infra.json](../backend/config/infra.json) — current deployment config
- [backend/processor/wiki_generator.py](../backend/processor/wiki_generator.py) — wiki orchestration (`_call_llm`, `generate_wiki`, `build_file_tree`, `read_readme`)
- [backend/promptstore/wiki_page.py](../backend/promptstore/wiki_page.py) — `WIKI_PAGE_CONTENT_PROMPT`, `build_wiki_page_prompt`, `WIKI_PAGE_REVIEW_PROMPT`, `format_page_catalog`
- [backend/promptstore/wiki_structure.py](../backend/promptstore/wiki_structure.py) — structure prompts
- [backend/modules/embedder/retriever.py](../backend/modules/embedder/retriever.py) — `call_with_file_filter`, `_call_with_file_filter_cloud`
- [backend/modules/embedder/code_splitter.py](../backend/modules/embedder/code_splitter.py) — `split_code_at_boundaries`, `build_enriched_chunk_text`
- [backend/clients/embedding_client.py](../backend/clients/embedding_client.py) — `SafeEmbedder`
- [backend/modules/chat/service.py](../backend/modules/chat/service.py) — `format_context_text`
- [backend/config/excluded.json](../backend/config/excluded.json), [backend/config/included.json](../backend/config/included.json), [backend/config/embedder.json](../backend/config/embedder.json)

Prior repo-memory notes worth reading before starting:
- `/memories/repo/embedding_stuck_huge_sql.md` — motivates Phase 2 size cap
- `/memories/repo/wiki_generation_pipeline_analysis.md` — broader pipeline notes
- `/memories/repo/codemap_complete_analysis.md`, `/memories/repo/codemap_wiki_integration_impact.md` — codemap context for Phase 5 step 4

---

*End of plan.*
