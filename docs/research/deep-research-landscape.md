# Deep Research Product Landscape (2024-2026)

A research brief for a developer building a "deep research" AI assistant. Covers the major products, the architecture patterns they share, the building blocks (APIs and tools) a builder would compose today, the architectural choices emerging as best practice, and the real risks you have to engineer around. Focus is on what exists *now* (as of 2026-09-22). Primary sources only.

---

## 1. Landscape — Major Products

### 1.1 OpenAI Deep Research

- **Launched**: 2025-02-02 inside ChatGPT. Initially gated to ChatGPT Pro ($200/mo); expanded to all paid tiers shortly after ([OpenAI: Introducing Deep Research](https://openai.com/index/introducing-deep-research/), [help.openai.com deep-research-faq](https://help.openai.com/en/articles/11369840-deep-research-faq)).
- **Model**: An early version of OpenAI o3 fine-tuned / RL-trained for web browsing ([Deep Research System Card, PDF](https://cdn.openai.com/deep-research-system-card.pdf)). The sidebar summary is produced by a separate o3-mini call that summarizes the chain-of-thought for the user.
- **Runtime**: Typically 5-30 minutes. A single run can involve dozens of searches and page fetches.
- **Tools available to the agent**: web search, browser (click/scroll), Python sandbox (for calculation/analysis/plotting), file parser (PDFs, images). The Python tool was added because many research questions require computation rather than retrieval.
- **Training method** (from the system card): reinforcement learning on browsing tasks. Mix of objective, auto-gradable tasks with ground truth and open-ended tasks with rubrics. A chain-of-thought model was used as a grader. The model inherits o1 safety datasets and adds new browsing-specific ones.
- **Citations**: Inline numeric citations tied to a sources panel. Each cited claim is anchored to a retrieved URL.
- **Access**: Web UI in ChatGPT. No developer API (as of this writing) — although OpenAI shipped an "Agent Mode" inside ChatGPT later, it is not a separate Deep Research API.
- **Benchmark**: 26.6% on Humanity's Last Exam, vs. 9.1% (o1), 9.4% (DeepSeek R1), 3.3% (GPT-4o) ([System Card](https://cdn.openai.com/deep-research-system-card.pdf)).
- **Risks the lab calls out**: prompt injection via web content, disallowed/regulated advice, privacy leakage from public data, ability to run code in the Python sandbox, bias, hallucinations. All Preparedness categories rated Medium after mitigation.

### 1.2 Perplexity Deep Research and Sonar

- **Sonar API**: Launched 2025-01-21 as Perplexity's developer-facing API for AI search ([Perplexity blog: Introducing Sonar](https://www.perplexity.ai/blog/introducing-sonar)). Sonar is a fine-tuned variant of Llama 3.3 70B for answer-engine tasks. Sonar Pro adds reasoning.
- **Deep Research**: Launched 2025-02 as a consumer-facing feature; later exposed via the Sonar API as `sonar-deep-research`. Free for all Perplexity users; Pro/Enterprise users get higher throughput and longer reports ([Perplexity FAQ: how does Deep Research work](https://www.perplexity.ai/hub/faq/how-does-perplexity-deep-research-work)).
- **Architecture**: A "supervisor" model orchestrates the Deep Research process, spawning many parallel attempts at sub-questions, deduplicating findings, and then writing the report ([Perplexity Sonar Architecture](https://www.perplexity.ai/hub/faq/explain-the-sonar-architecture)). Citation style is inline footnote links.
- **Access**: Web UI at perplexity.ai (free), Pro tier for faster runs, Sonar API for developers ([docs.perplexity.ai/.../sonar-api](https://docs.perplexity.ai/en/home/sonar-api)).

### 1.3 Google Gemini Deep Research

- **Launched**: 2024-12-11 inside Gemini Advanced ([TechCrunch: Gemini can now research deeper](https://techcrunch.com/2024/12/11/gemini-can-now-research-deeper/)). Major 2025-12-11 rework under Gemini 3 Pro.
- **Public developer API**: 2026-04-21 via the Interactions API ([Google AI for Developers: Gemini Deep Research](https://ai.google.dev/gemini-api/docs/deep-research), [tokencost.app breakdown](https://tokencost.app/blog/gemini-deep-research-agent-cost)). The Interactions API went GA on 2026-06-22 ([Interactions API GA](https://ai-watch-blog.vercel.app/en/posts/2026-06-22-gemini-interactions-api-ga)).
- **Architecture** (from Google's own docs and coverage): three-stage pipeline — **Planner → Browser → Synthesizer**. Planner decomposes the question into a multi-step research plan; the Browser executes Google searches and follows links into Workspace sources (Gmail, Drive, Chat, BigQuery via MCP) when permitted; the Synthesizer compiles a 15+ page cited report with multiple self-critique passes ([blog.google: Build with Gemini Deep Research](https://blog.google/technology/developers/deep-research-agent-gemini-api/)).
- **Runtime**: Up to ~60 minutes of autonomous execution. Must be called async (`background=true`).
- **Models and pricing** ([tokencost.app](https://tokencost.app/blog/gemini-deep-research-agent-cost), [nowosci.ai](https://www.nowosci.ai/en/article/google-deep-research-max-api-agent)):
  - `deep-research-preview-04-2026` — ~80 queries, $1-3/task
  - `deep-research-max-preview-04-2026` — ~160 queries, $3-7/task
  - Plus Google Search grounding at $14 per 1,000 queries (enabled by default)
  - Base Gemini 3 Pro: $2/$12 per 1M tokens (≤200k context); cached input $0.20/1M
- **Benchmarks**: 46.4% HLE, 59.2% BrowseComp, 66.1% DeepSearchQA. ~40% hallucination reduction vs. base Gemini.
- **Open source alongside**: Google released **DeepSearchQA** as an open benchmark — 900 causal-chain tasks across 17 fields.

### 1.4 Anthropic — Claude with web tools

Anthropic does **not** ship a "Deep Research" branded product as of 2026-09-22. What they ship instead are **server-side tools** that you compose into a deep research workflow yourself:

- **`web_search` tool** — launched 2025-05-07 ([anthropic.com/news/web-search-api](https://www.anthropic.com/news/web-search-api)). Three versions: `web_search_20250305`, `web_search_20260209` (adds dynamic filtering via code execution), `web_search_20260318` (adds `response_inclusion` for agentic workflows). Pricing **$10 per 1,000 searches** plus tokens ([platform.claude.com: web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)).
- **`web_fetch` tool** — launched 2025-09-10. Returns full text content (and base64 PDFs) from a URL. Citations are optional (`citations.enabled: true`). Versions track dynamic filtering, `use_cache` bypass, and `response_inclusion`. **No additional charge** beyond tokens ([platform.claude.com: web fetch tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool)).
- **Combined behavior**: When both tools are enabled and the user names a page without a URL, Claude searches first, then fetches. Dynamic filtering (2026 versions) lets Claude write and run code that filters search/fetch results before they hit context, which materially reduces token cost on search-heavy research.
- **Constraints worth knowing**: `web_fetch` cannot render JS-heavy pages ([platform.claude.com: web fetch tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool) explicitly directs you to the browser-use tool for that). `web_fetch` also enforces URL validation — Claude can only fetch URLs that have appeared elsewhere in the conversation (user messages, prior tool results), not URLs that appear only in Claude's own output, which limits exfiltration and prompt-injection risk.
- **Practical implication for builders**: a Claude-based deep research product is essentially the agent loop **you** write, with Claude's web tools as primitives. Anthropic also ships **Managed Agents** ([platform.claude.com: Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview)) which can host a longer-running agent with per-tool domain restrictions.

### 1.5 Hugging Face Open Deep Research (smolagents)

- **What it is**: Not a product — an **open-source reproduction** of OpenAI's Deep Research, built in a 24-hour sprint by 5 HF engineers in February 2025, living at [`huggingface/smolagents/examples/open_deep_research/`](https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research).
- **Architecture (multi-agent)** ([HF blog: Open Deep Research](https://huggingface.co/blog/open-deep-research), README):
  - **Manager agent** — a `CodeAgent` (writes actions as Python, not JSON), max 12 steps, plans every N steps.
  - **Search sub-agent** — a `ToolCallingAgent` named `search_agent`, max 20 steps, plans every 4 steps. The manager delegates to it.
  - **Why code-based actions**: per [Wang et al. 2024 (arXiv:2402.01030)](https://arxiv.org/abs/2402.01030), CodeAgent uses ~30% fewer steps than JSON-tool-call agents because parallel/sequential actions collapse into a single code block. State (e.g., downloaded images) can be stored in variables.
- **Tools it ships with** (largely borrowed from Microsoft's Magentic-One):
  - `web_search` — SerpApi (Google SERP) by default; configurable to Serper.dev.
  - `visit_page` — text-only browser with markdownify stripping scripts/styles; viewport of 5120 chars.
  - `download_file`, `find_archived_url` (Wayback), `PageUp/PageDown`, `find_on_page_ctrl_f`.
  - `inspect_file_as_text` — text inspector for HTML, PDF, DOCX, XLSX, PPTX, audio (via speech_recognition), images (OCR + vision caption).
- **Citations**: implicit. No structured citation tracker. The reformulation step forces a strict `FINAL ANSWER:` format for the GAIA benchmark, but inline citations are inherited only if the LLM chooses to insert source URLs.
- **Benchmark**: 55.15% on GAIA validation, vs. OpenAI Deep Research's 67.36% and a plain smolagents baseline of ~33%.

### 1.6 Other 2025-2026 entrants

- **xAI Grok DeepSearch** — launched 2025-02 with Grok 3 ([x.ai/blog/grok-3-beta](https://x.ai/blog/grok-3-beta)). Architecture is a tool-augmented agent on top of Grok 3: query planner → parallel web/X retrievers → evidence aggregator → Grok 3 "Think" mode synthesizes the cited answer. Iteratively refines queries. Available to SuperGrok and X Premium+ subscribers via grok.com and the X app.
- **You.com** — You.com has shipped agent-style research features since 2024 (Research agent, ARI), built on their hybrid search/answer stack. ([you.com](https://you.com))
- **Mistral Le Chat Deep Research** — Mistral has shipped a "deep research" mode in Le Chat; underlying model family is Mistral's own (Mixtral/Mistral Large). ([mistral.ai](https://mistral.ai))
- **DeepSeek DeepResearch** — DeepSeek released a Deep Research agent in their chat product (DeepSeek-V3.1 / R1 lineage) ([deepseek.com](https://deepseek.com)). Notable for an open-weight backbone and aggressive pricing.
- **Genspark** — AI search engine producing cited answer pages; shifted in 2025 toward agent-style multi-page reports.
- **Felo / Skywork / Manus / Genspark / etc.** — a long tail of 2025 entrants, all variants on the "agent + many searches + cited report" template. None has displaced the top tier as of mid-2026.

---

## 2. Common Architecture Patterns

Every shipping deep research system is, at its core, the same thing: a reasoning model in a tool-use loop. Below is the canonical inner loop, with citations to who does what.

### 2.1 Query decomposition / planning

- **OpenAI Deep Research** — implicit, in-chain. The system card describes the model as "decomposing" the user's task into sub-questions and deciding what to search for next ([System Card](https://cdn.openai.com/deep-research-system-card.pdf)). It also asks the user clarifying questions about the desired scope, time frame, and source preferences.
- **Gemini Deep Research** — explicit planner. Google's docs describe the **Planner** as a separate stage that produces a multi-step research plan and surfaces it for user review/modification before running ([blog.google](https://blog.google/technology/developers/deep-research-agent-gemini-api/)).
- **Hugging Face Open Deep Research** — the manager `CodeAgent` plans a research trajectory in Python, then delegates per-subtask to the search sub-agent ([README](https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research)).
- **Smolagents in general** — plans every N steps (configurable), writing the plan as a Python comment block in the next action.

### 2.2 Search-query generation (how many, how refined)

- **OpenAI** — dozens to ~hundreds of queries per run. The model refines queries based on what it learns and **can backtrack** ([System Card](https://cdn.openai.com/deep-research-system-card.pdf)).
- **Perplexity** — the supervisor runs "many things in parallel" for the same sub-question, then deduplicates ([Perplexity Sonar architecture FAQ](https://www.perplexity.ai/hub/faq/explain-the-sonar-architecture)).
- **Gemini** — also uses parallel search; Google's docs describe the Browser stage as iterating search → parse → identify gaps → search again.
- **Open Deep Research** — the search sub-agent has a hard cap of 20 steps (so up to 20 search/browse actions).

### 2.3 Web fetching vs search-API patterns

There are essentially two patterns:

1. **Search API → URL → fetch**. Search returns a ranked URL list; the agent fetches the full page (HTTP GET, possibly via a headless browser for JS). Most production systems follow this. Tools used at each layer are configurable (see §3).
2. **Native server-side web tools**. Anthropic's `web_search` + `web_fetch` and OpenAI's Responses API `web_search` tool collapse both into a single provider-managed call. The trade-off is less control but less engineering.

### 2.4 Source ranking, filtering, deduplication

- **Open Deep Research** has no formal dedup — relies on the LLM.
- **Perplexity** explicitly runs parallel retrieval attempts and dedupes by content hash before synthesis.
- **Anthropic's dynamic filtering** (2026 versions) is a notable new pattern: instead of dumping every search result into context, Claude writes and executes a filter that drops irrelevant pages *before* they hit the token budget ([web search tool docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)). Materially cheaper than naive retrieval for multi-step research.

### 2.5 Reading and note-taking

Every system summarizes retrieved pages, but how they store the summaries varies:

- **Inline in context** (OpenAI Deep Research, Anthropic Claude, Open Deep Research) — the agent's conversation history *is* the notebook.
- **Explicit scratchpad** — Gemini's Planner keeps a structured plan that survives rewrites.
- **Files on disk / sandboxed filesystem** — Open Deep Research has the manager download files (`.xlsx`, `.pdf`, `.png`) and re-inspect them with `inspect_file_as_text`; this lets it handle multimodal input via tools rather than dumping bytes into the prompt.

### 2.6 Synthesis

A second "writer" pass is universal. Examples:

- **OpenAI** — the model itself generates the final report in the same loop; o3-mini is then called *after* to summarize the chain-of-thought for the sidebar ([System Card](https://cdn.openai.com/deep-research-system-card.pdf)).
- **Gemini** — Synthesizer stage explicitly performs multiple self-critique passes.
- **Open Deep Research** — the `reformulator.py` post-step prompts the model with the original task + transcript and forces a strict `FINAL ANSWER: ...` extraction, originally for GAIA-style exact-answer grading.

### 2.7 Citation handling

This is where systems diverge most:

- **Inline numbered citations** (OpenAI, Perplexity, Gemini) — cleanest UX.
- **Structured citations with character offsets** (Anthropic `web_fetch` with `char_location` blocks) — most rigorous; each citation carries a `cited_text` snippet and exact span.
- **Implicit, opportunistic** (Open Deep Research) — the model decides whether to mention a source. Unreliable for production.

### 2.8 Iterative depth and stop conditions

- **OpenAI** runs until the model emits a final answer; depth bounded by time and step count.
- **Gemini** uses an "async task manager" that maintains shared state between Planner and task models, allowing graceful recovery and step-bounded loops ([Google blog](https://blog.google/technology/developers/deep-research-agent-gemini-api/)).
- **Open Deep Research** hard-caps the manager at 12 steps and the search sub-agent at 20 steps.
- **Anthropic** lets you set `max_uses` per tool (e.g., `max_uses: 10` for web fetch) — a hard cap that doubles as a cost control.

### 2.9 Multi-agent vs single-agent

- **Single-agent with planning** is the default: OpenAI Deep Research, Claude with web tools, Perplexity Deep Research, Gemini Deep Research (Planner/Browser/Synthesizer are stages inside one orchestration).
- **Hierarchical multi-agent** is rarer but exists: Open Deep Research's manager+search-agent split. LangChain's own `open_deep_research` has historically oscillated between multi-agent and single-loop designs, and notably **the single-loop variant now beats the multi-agent one on DeepResearch Bench** ([dreaming.press overview](https://dreaming.press/posts/open-source-deep-research-agents.html)).
- **General rule emerging in 2026**: the marginal benefit of multi-agent diminishes when the underlying model is strong enough to plan by itself. Multi-agent still helps when each agent needs different tools or different system prompts (e.g., one writes code, one searches the web).

---

## 3. Building Blocks — APIs and Tools

Pricing is per the vendor's published page at the time of writing (mid-2026); always confirm before budgeting.

### 3.1 Search APIs

| Vendor | What it is | Pricing (mid-2026) | Notes |
|---|---|---|---|
| **Tavily** ([tavily.com](https://tavily.com)) | Search + extract in one API, designed for agents | Free: 1,000 credits/mo. Project $30/4k, Startup $220/38k, Growth $500/100k. Basic search = 1 credit, Advanced = 2 credits. | Built-in content extraction (saves a round-trip). 93.3% on SimpleQA. ([docs.tavily.com](https://docs.tavily.com)) |
| **Exa** ([exa.ai](https://exa.ai)) | Neural / semantic search over its own index | $20 signup credit + $10/mo free. Starter $49, Pro $449. Per-request: neural 1-25 results $5/1k, neural 26-100 $25/1k; **Exa Deep $12-15/1k**. | Claims ~100B docs indexed; Exa Deep runs multi-search with ranking ([exa.ai/changelog](https://exa.ai/docs/changelog)). |
| **Brave Search API** ([brave.com/search/api](https://brave.com/search/api/)) | Independent web index | Free: 2,000 queries/mo @ 1 QPS. Pro AI: $3-5/CPM; Pro Data: $5-9/CPM. | No AI re-processing on Data tier; AI tier adds summarization. |
| **Serper** ([serper.dev](https://serper.dev)) | Google SERP API | Free: 2,500 queries (one-time). Starter $50/50k, Growth $130/200k. ~$0.30/1k at scale. | Cheap, fast, Google-shaped results. The default in HF's Open Deep Research. |
| **Google Programmable Search Engine (CSE)** ([developers.google.com/custom-search](https://developers.google.com/custom-search)) | Official Google site-restricted search | Free first 100/day, then $5/1k (max 10k/day). | Limited and expensive; mostly useful when you *must* use Google's index for compliance reasons. |
| **Bing Web Search API** ([learn.microsoft.com/bing/search](https://learn.microsoft.com/en-us/bing/apis/bing-web-search-api)) | Microsoft Bing index | Free 1k/mo with Azure. S1 $3/1k (up to 250k/mo), S2 $7/1k, S3 custom. | Standard for Microsoft-stack customers. |
| **Perplexity Sonar** | Already covered | $5/1k searches (matches Exa neural) | The "answer engine" layer above a search API. |

**Tactical advice**: for an MVP, start with **Tavily** (cheap, integrated extract) or **Exa** (neural quality). Add **Serper** for high-volume cheap SERP data behind the scenes. Keep **Brave** as a privacy-friendly second index.

### 3.2 Web fetching and scraping

| Vendor | What it is | Pricing | Notes |
|---|---|---|---|
| **Firecrawl** ([firecrawl.dev](https://firecrawl.dev)) | Managed scraping + markdown conversion + crawl | Free: 500 credits. Hobby ~$19/3k, Standard ~$99/25k, Pro ~$299/100k, Enterprise custom. ([firecrawl.dev/pricing](https://firecrawl.dev/pricing)) | Five operations: Scrape, Crawl, Extract (LLM-powered JSON), Map (URL discovery), Search. JS rendering, stealth proxies. |
| **Jina Reader** ([jina.ai/reader](https://jina.ai/reader)) | `r.jina.ai/{URL}` returns LLM-friendly markdown; `s.jina.ai/{query}` does search | Free, 200 RPM, no API key required. | Headers control behavior: `X-Target-Selector`, `X-Wait-For-Selector`, `X-Respond-With`. PDFs supported. Best free tier in the category. |
| **Playwright** ([playwright.dev](https://playwright.dev)) | Open-source browser automation | Free; you run the browsers | The default for JS-heavy sites and authenticated flows. |
| **Browserbase** ([browserbase.com](https://browserbase.com)) | Managed Playwright/Puppeteer/Selenium | Free ~1 hr. Developer ~$20/10 hr. Production custom. | Stealth mode, captcha solving, 100+ countries. Pairs with LangChain and CrewAI. |
| **Browserless, Steel.dev, Anchor Browser, Hyperbrowser** | Alternatives | Varies | Browserbase is currently the most-adopted in the agent ecosystem. |
| **Anthropic `web_fetch`** | Provider-managed | No additional fee | Limited to text/HTML/PDF, no JS. URL validation prevents exfiltration. |
| **OpenAI `web_search` tool** (Responses API) | Provider-managed | See OpenAI pricing | Bundled inside the Responses API. |

**Tactical advice**: a typical deep research stack uses **Jina Reader** for the cheap 80% of pages and **Firecrawl or Browserbase** for the long tail of JS-rendered or anti-bot-protected pages. Anthropic's `web_fetch` is best when you've already filtered to high-value URLs (it has no JS).

### 3.3 LLM APIs with native web tools

| Provider | Tool | Pricing | Notable |
|---|---|---|---|
| **Anthropic Claude** | `web_search_20260318`, `web_fetch_20260318` | web_search: $10/1k; web_fetch: free (token only) | Dynamic filtering via code execution; URL validation; `response_inclusion` for token savings. ([docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)) |
| **OpenAI** | Responses API `web_search` tool | Per OpenAI's published tool pricing | Bundled into Chat Completions / Responses; no separate Deep Research API as of writing. |
| **Google Gemini** | Google Search grounding | $14 per 1,000 grounding queries | Enabled by default on Deep Research. ([ai.google.dev/gemini-api/docs/deep-research](https://ai.google.dev/gemini-api/docs/deep-research)) |

### 3.4 Agent frameworks

- **LangGraph** ([github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)) — graph-based, stateful agent loops. Wins on deterministic, long-running, human-in-the-loop research with checkpoints. Has a public **Open Deep Research** template (`langchain-ai/open_deep_research`) that explicitly dropped a prior supervisor-researcher multi-agent design in favor of a single-loop variant after benchmarking on DeepResearch Bench.
- **smolagents** ([github.com/huggingface/smolagents](https://github.com/huggingface/smolagents)) — `CodeAgent` writes actions as Python, ~30% fewer steps than JSON agents. ~1,000 lines of core code. Native Open Deep Research example included. Sandbox options: E2B, Blaxel, Modal, Docker (LocalPythonExecutor is explicitly not a security boundary).
- **CrewAI** ([github.com/crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)) — role-based "crew" of agents. Best for fast prototyping where each agent has a clear persona (Researcher, Writer, Critic).
- **AutoGen** ([github.com/microsoft/autogen](https://github.com/microsoft/autogen)) — Microsoft Research. Conversable agents with code execution; v0.4+ (2024-2025) introduced an async event-driven architecture. Magentic-One, the multi-agent research orchestrator on which HF's Open Deep Research leans, ships in this lineage.
- **Pydantic AI** ([github.com/pydantic/pydantic-ai](https://github.com/pydantic/pydantic-ai)) — type-safe, Pydantic-native. Strong where structured outputs and validation matter.
- **OpenAI Agents SDK** ([github.com/openai/openai-agents-python](https://github.com/openai/openai-agents-python)) — official lightweight SDK for OpenAI's tool-calling models; handoffs, tracing, sessions.
- **Google ADK** ([github.com/google/adk-python](https://github.com/google/adk-python)) — Agent Development Kit, released 2025; aligns with the Interactions API.

### 3.5 RAG / vector components

Standard pgvector/Qdrant/Weaviate are useful as **post-fetch storage** when you want to resume a long-running research session, share a corpus across users, or cache results. None of the major hosted deep research products expose a vector DB by default — they keep everything in the model's context window for the duration of the run. For multi-session research, plan on adding a vector store with metadata (URL, timestamp, query that surfaced it) so subsequent runs can short-circuit.

---

## 4. Concrete Architectural Choices for the Builder

These are the patterns emerging as consensus across the products and frameworks surveyed. Be opinionated where the evidence supports it.

### 4.1 Sequential vs parallel research subagents

**Parallel wins when (a) the sub-questions are independent, (b) the underlying LLM is strong enough to deduplicate, and (c) latency matters.** Perplexity runs many parallel attempts at the same sub-question, then deduplicates by content hash. Gemini's Browser stage spawns parallel fetches across pages.

**Sequential wins when the next question depends on the previous answer** (e.g., "what does the top result say about X?"). Pure parallelism on dependent questions produces redundant fetches.

**Practical default for a v1**: One manager that runs a tight sequential loop, with **parallel fetches inside each step** (one search → N parallel `web_fetch` calls → one synthesis step). Promote to multi-agent only when you've measured the single-loop bottleneck.

### 4.2 Single-pass vs iterative deepening

Single-pass ("generate all queries, fetch all pages, write the report") is fast but shallow. Iterative deepening ("read, identify gaps, re-search") catches missing sources and contradictions but multiplies cost.

- **OpenAI** and **Gemini** are strongly iterative (5-30 min / up to 60 min respectively).
- **Open Deep Research** has hard step caps (manager 12, search 20) that force a bounded depth.
- **Anthropic's `max_uses`** lets you cap `web_search` and `web_fetch` separately per request — a clean way to enforce depth budgets.

**Practical default**: an iterative loop with hard caps. Use **prompted stopping criteria** ("if you have 3+ independent sources confirming a claim, move on") and a **budget circuit breaker** (max steps, max wall-clock, max dollars per run).

### 4.3 Citation grounding

This is the single biggest quality lever. Three patterns worth combining:

1. **Native citations from the search/fetch tool.** Anthropic's `web_fetch` returns `char_location` blocks with the exact cited text span; Gemini returns grounding chunks with source attribution; OpenAI's Deep Research exposes a sources panel. Use these — don't reconstruct citations by string-matching URLs back into the LLM output.
2. **Bind citations at write time, not after.** Gemini's Synthesizer stages this explicitly; the citation is part of the writing step so it survives rewrites. If you post-process to insert citations, you will get citation drift (claims that don't match the cited source).
3. **Verifiable snippets.** Where you control the pipeline (e.g., Tavily or Firecrawl + your own LLM), include the actual text snippet the citation claims to support. This makes post-hoc verification tractable.

### 4.4 Bounding cost and latency

Real-world cost shapes:

- **OpenAI Deep Research** consumes dozens of o3 calls and dozens of web fetches per run. Equivalent to $5-50 of inference per report depending on depth.
- **Gemini Deep Research API** is $1-7/task plus $14/1k grounding queries ([tokencost.app](https://tokencost.app/blog/gemini-deep-research-agent-cost)).
- **Anthropic** charges $10/1k searches plus token costs; `web_fetch` is free but the tokens add up.

Engineering controls:

- **Hard step caps per agent** (Open Deep Research style).
- **Per-tool `max_uses`** (Anthropic style).
- **Wall-clock and dollar budgets per run**, surfaced to the user upfront ("this will run for up to X minutes and cost up to $Y").
- **Prompt caching** for static tool definitions and repeated research plans (Anthropic supports this; OpenAI and Google have their equivalents).
- **Dynamic filtering** (Anthropic 2026 tool versions) to keep irrelevant search/fetch content out of the token bill.
- **Streaming the report as it's written** rather than only at the end — this is what every shipping product does.

### 4.5 Evaluation

Benchmarks and harnesses to track:

- **GAIA** — multi-modal, tool-using agent benchmark. The headline number for "deep research" style agents ([Hugging Face: GAIA-annotated](https://huggingface.co/datasets/smolagents/GAIA-annotated)).
- **Humanity's Last Exam (HLE)** — reasoning benchmark. OpenAI Deep Research hit 26.6%; Gemini Deep Research 46.4%.
- **BrowseComp** — browsing-specific benchmark. Gemini Deep Research 59.2%.
- **DeepSearchQA** — Google's open release alongside their Deep Research agent: 900 causal-chain tasks across 17 domains ([blog.google](https://blog.google/technology/developers/deep-research-agent-gemini-api/)).
- **DeepResearchBench** — what LangChain's `open_deep_research` used to show that the single-loop design beats their older multi-agent one.

For internal eval, build a small eval set of (a) questions with known-good answers and (b) questions with deliberately tricky citation traps. Measure:

1. **Citation validity** (does the URL resolve to a real page?)
2. **Entailment** (does the cited page actually support the claim?)
3. **Coverage** (do you have ≥2 independent sources for decision-critical claims?)
4. **Faithfulness under truncation** (does the report survive when you cut context mid-run?)

---

## 5. Risks and Gotchas

The published literature on deep-research-agent failures is more useful than generic "LLM hallucination" advice. Key references: [arXiv 2604.03173](https://arxiv-vanity.com/papers/2604.03173) (Reference Hallucinations in Commercial LLMs and Deep Research Agents), [arXiv 2608.28643](https://arxiv.org/html/2608.28643) (Faithful Reports), [LinkedIn: Up to Half of Your AI Research Assistant's Citations Are Fake](https://www.linkedin.com/pulse/up-half-your-ai-research-assistants-citations-fake-real-bilal-nm7uf), and the [fieldway.org](https://fieldway.org/blog/research-agents-get-less-accurate-the-more-they-read-42-worse-from-2-tool-calls-to-150) analysis of accuracy drift with tool-call count.

### 5.1 Hallucinated sources and citation drift

- **3-13% of citation URLs are hallucinated** across commercial systems; **5-18% don't resolve**. Deep research agents have the highest fabrication rates despite producing the most citations.
- **Statement-level misattribution**: even when the URL is real, ~50% of OpenAI Deep Research's cited statements contained subtle inaccuracies; Gemini 2.5 Pro and Perplexity Deep Research have ~47-50% fabricated references in some domains.
- **Citation drift with tool-call depth**: fact-check accuracy drops ~42% as tool calls scale from 2 to 150. Link validity stays high (>92%); the degradation is in *synthesis*, not *source selection*. GPT-5.4: 79% → 17%; Claude Opus 4.6: 80% → 58%.
- **Multi-turn drift**: 29-86% of citations mutate between conversation turns.

**Mitigations**: bind citations at write-time; include the cited snippet in the citation metadata; never let the model cite URLs it hasn't actually fetched; verify URLs server-side after the run.

### 5.2 Paywalls and dynamic content

- Most search APIs only return what's crawlable; paywalled content (NYT, academic journals) is invisible.
- JS-rendered pages (SPAs) often need a headless browser. Anthropic's `web_fetch` explicitly does not render JS — pair with Playwright or Browserbase for those.
- Dynamic prices, live scores, and rapidly-changing pages may need `use_cache: false` (Anthropic 2026 tool versions) or Brave's freshness controls.

### 5.3 Rate limits and cost overruns

- Long-running research can blow past rate limits silently — a 30-minute run doing 200 web fetches is well over most vendors' per-second ceilings.
- Build a **budget guard**: track dollars and tool calls, abort gracefully when exceeded, surface a clear "I stopped because I ran out of budget" message.
- Tavily dev tier caps at 100 RPM; production at 1,000. Exa Starter tier is generous but Exa Deep at $12-15/1k adds up fast. Anthropic at $10/1k searches means a 100-search run is $1 of search cost on top of tokens.

### 5.4 Anti-bot protections

Cloudflare, DataDome, and Akamai block naive scraping. Firecrawl's stealth mode, Browserbase's stealth mode, and residential proxy networks help but are not free and not bulletproof. Build graceful failure: if a page returns a 403 or a "verify you are human" page, the agent should fall back to (a) cached/archived versions (Wayback Machine — `find_archived_url` in Open Deep Research), (b) alternative sources, or (c) admitting it can't access the page.

### 5.5 Prompt injection from web content

The OpenAI system card explicitly calls this out as a top risk. Anthropic's `web_fetch` enforces URL validation (Claude can only fetch URLs that appeared in earlier conversation context) precisely to limit injection-via-fetch. Defenses: sandbox code execution; treat retrieved text as untrusted; never let fetched content reach privileged tool calls without a model-side judgment step. [DeepResearchGuard (ACL 2026)](https://hyper.ai/en/papers/acl__2026__2026.acl-long.2010) is the most thorough published framework here, with a four-stage safeguard pipeline.

### 5.6 Long-horizon drift

As cited above, the longer the agent runs, the worse its synthesis gets — even when source selection stays accurate. Counter-intuitively, a single-pass report may be more faithful than a 30-minute research marathon. Practical implication: **don't optimize for more steps; optimize for fewer, better ones.** Parallelize where possible; cap depth; commit to citations as you write.

### 5.7 Eval blind spot

Most published benchmarks (GAIA, HLE, BrowseComp) measure "did the agent get the right answer" but not "did it cite accurately and faithfully." For production, you need a citation-fidelity eval in addition to answer-correctness eval. [ClaimProbe / ClaimWriter](https://arxiv.org/abs/...) and [DeepTRACE](https://liner.com/review/deeptrace-auditing-deep-research-ai-systems-for-tracking-reliability) from Salesforce are the leading published harnesses.

---

## Sources Index

**Official product pages and system cards**
- OpenAI: [Introducing Deep Research](https://openai.com/index/introducing-deep-research/)
- OpenAI: [Deep Research System Card (PDF)](https://cdn.openai.com/deep-research-system-card.pdf)
- OpenAI: [Help Center — Deep Research FAQ](https://help.openai.com/en/articles/11369840-deep-research-faq)
- Perplexity: [Sonar API documentation](https://docs.perplexity.ai/en/home/sonar-api)
- Perplexity: [Sonar Architecture FAQ](https://www.perplexity.ai/hub/faq/explain-the-sonar-architecture)
- Perplexity: [Deep Research FAQ](https://www.perplexity.ai/hub/faq/how-does-perplexity-deep-research-work)
- Google: [blog.google — Build with Gemini Deep Research](https://blog.google/technology/developers/deep-research-agent-gemini-api/)
- Google: [ai.google.dev — Gemini Deep Research](https://ai.google.dev/gemini-api/docs/deep-research)
- Google: [Interactions API GA announcement](https://ai-watch-blog.vercel.app/en/posts/2026-06-22-gemini-interactions-api-ga)
- Google: [Workspace blog — Create detailed reports with Deep Research](https://workspace.google.com/blog/ai-and-machine-learning/meet-deep-research-your-new-ai-research-assistant)
- Anthropic: [Introducing web search on the Anthropic API](https://www.anthropic.com/news/web-search-api)
- Anthropic: [Web search tool docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)
- Anthropic: [Web fetch tool docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool)
- Anthropic: [Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview)
- xAI: [Grok 3 Beta — Think and DeepSearch](https://x.ai/blog/grok-3-beta)
- Hugging Face: [Open Deep Research blog post](https://huggingface.co/blog/open-deep-research)
- Hugging Face: [smolagents repo](https://github.com/huggingface/smolagents)
- Hugging Face: [Open Deep Research README](https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research)
- Hugging Face: [GAIA-annotated dataset](https://huggingface.co/datasets/smolagents/GAIA-annotated)
- Microsoft: [Magentic-One (in autogen)](https://github.com/microsoft/autogen)
- LangChain: [open_deep_research](https://github.com/langchain-ai/langgraph) (see `examples/open_deep_research`)

**API vendor docs**
- Tavily: [docs.tavily.com](https://docs.tavily.com)
- Exa: [exa.ai/changelog](https://exa.ai/docs/changelog)
- Brave: [brave.com/search/api](https://brave.com/search/api/)
- Serper: [serper.dev](https://serper.dev)
- Google CSE: [developers.google.com/custom-search](https://developers.google.com/custom-search)
- Bing: [learn.microsoft.com/bing/search](https://learn.microsoft.com/en-us/bing/apis/bing-web-search-api)
- Firecrawl: [firecrawl.dev/pricing](https://firecrawl.dev/pricing)
- Jina: [jina.ai/reader](https://jina.ai/reader)
- Browserbase: [browserbase.com](https://browserbase.com)
- Playwright: [playwright.dev](https://playwright.dev)

**Evaluation and risk literature**
- arXiv: [Executable Code Actions Elicit Better LLM Agents (Wang et al., 2024)](https://arxiv.org/abs/2402.01030)
- arXiv: [Detecting and Correcting Reference Hallucinations](https://arxiv-vanity.com/papers/2604.03173)
- arXiv: [Faithful Reports (Redesigning Deep Research Writing)](https://arxiv.org/html/2608.28643)
- ACL 2026: [DeepResearchGuard](https://hyper.ai/en/papers/acl__2026__2026.acl-long.2010)
- fieldway.org: [Research Agents Get Less Accurate the More They Read](https://fieldway.org/blog/research-agents-get-less-accurate-the-more-they-read-42-worse-from-2-tool-calls-to-150)
- LinkedIn: [Up to Half of Your AI Research Assistant's Citations Are Fake](https://www.linkedin.com/pulse/up-half-your-ai-research-assistants-citations-fake-real-bilal-nm7uf)
- DeepTRACE: [Liner summary](https://liner.com/review/deeptrace-auditing-deep-research-ai-systems-for-tracking-reliability)

**Pricing and coverage references**
- tokencost.app: [Gemini Deep Research cost breakdown](https://tokencost.app/blog/gemini-deep-research-agent-cost)
- nowosci.ai: [Google Deep Research Max](https://www.nowosci.ai/en/article/google-deep-research-max-api-agent)
- TechCrunch: [Gemini can now research deeper (Dec 2024)](https://techcrunch.com/2024/12/11/gemini-can-now-research-deeper/)
- dreaming.press: [Open-Source Deep Research Agents overview](https://dreaming.press/posts/open-source-deep-research-agents.html)