# Deep Research 产品全景（2024–2026）

> 面向正在搭建"deep research"（深度研究）类 AI 助手的开发者。覆盖主要产品、它们共有的架构模式、今日可直接组合使用的 API / 工具、正在成为最佳实践的架构选择，以及真正必须在工程上规避的风险。重点是"当下"（2026-09-22）真实存在的事物。**仅引用一手来源（primary sources）。**

---

## 1. 产品全景——主要玩家

### 1.1 OpenAI Deep Research

- **上线时间**：2025-02-02，在 ChatGPT 内部上线。最初仅对 ChatGPT Pro（$200/月）开放；不久后扩展到所有付费档位（[OpenAI: Introducing Deep Research](https://openai.com/index/introducing-deep-research/)、[help.openai.com deep-research-faq](https://help.openai.com/en/articles/11369840-deep-research-faq)）。
- **模型**：在 OpenAI o3 的早期版本上针对网页浏览做了微调 / RL 训练（[Deep Research System Card, PDF](https://cdn.openai.com/deep-research-system-card.pdf)）。侧边栏摘要由一次单独的 o3-mini 调用生成，用于把 chain-of-thought 摘要给用户看。
- **运行时长**：通常 5–30 分钟。单次运行可能涉及数十次搜索与网页抓取。
- **Agent 可用工具**：网页搜索、浏览器（点击 / 滚动）、Python 沙箱（用于计算 / 分析 / 绘图）、文件解析器（PDF、图片）。加入 Python 工具的原因是很多研究问题本质上是计算而非检索。
- **训练方法**（来自 System Card）：在浏览任务上进行强化学习。混合使用客观的、可自动打分的、带 ground truth 的任务，以及带评分细则（rubric）的开放式任务。一个 chain-of-thought 模型被用作打分器。该模型继承了 o1 的安全数据集，并新增了与浏览相关的安全数据。
- **引用形式**：行内数字编号引用，绑定到一个"来源"面板。每一条被引用的论断都锚定到一个被检索过的 URL。
- **访问入口**：ChatGPT Web UI。截至撰写本文时**未提供开发者 API**——虽然 OpenAI 后来在 ChatGPT 内上线了"Agent Mode"，但它并不是一个独立的 Deep Research API。
- **基准**：Humanity's Last Exam（HLE）26.6%，对比 o1（9.1%）、DeepSeek R1（9.4%）、GPT-4o（3.3%）（[System Card](https://cdn.openai.com/deep-research-system-card.pdf)）。
- **官方点名的风险**：来自网页内容的提示词注入（prompt injection）、违规 / 受监管的建议、来自公开数据的隐私泄露、Python 沙箱中执行代码的能力、偏见、幻觉。所有 Preparedness 类别在 mitigation 后评级为 Medium。

### 1.2 Perplexity Deep Research 与 Sonar

- **Sonar API**：2025-01-21 上线，作为 Perplexity 面向开发者的 AI 搜索 API（[Perplexity blog: Introducing Sonar](https://www.perplexity.ai/blog/introducing-sonar)）。Sonar 是为"答案引擎"任务微调的 Llama 3.3 70B 变体。Sonar Pro 额外增加了推理能力。
- **Deep Research**：2025-02 作为面向消费者的功能上线；之后通过 Sonar API 以 `sonar-deep-research` 形式开放给开发者。对所有 Perplexity 用户免费；Pro / Enterprise 用户享有更高吞吐和更长报告（[Perplexity FAQ: how does Deep Research work](https://www.perplexity.ai/hub/faq/how-does-perplexity-deep-research-work)）。
- **架构**：一个"监督者（supervisor）"模型负责编排整个 Deep Research 流程——为子问题并行生成大量尝试、对结果去重，然后撰写最终报告（[Perplexity Sonar Architecture](https://www.perplexity.ai/hub/faq/explain-the-sonar-architecture)）。引用样式为行内脚注链接。
- **访问入口**：perplexity.ai Web UI（免费）、Pro 档位享有更快运行速度、Sonar API 面向开发者（[docs.perplexity.ai/.../sonar-api](https://docs.perplexity.ai/en/home/sonar-api)）。

### 1.3 Google Gemini Deep Research

- **上线时间**：2024-12-11，在 Gemini Advanced 内上线（[TechCrunch: Gemini can now research deeper](https://techcrunch.com/2024/12/11/gemini-can-now-research-deeper/)）。2025-12-11 在 Gemini 3 Pro 之下做了重大重做。
- **公开开发者 API**：2026-04-21 通过 Interactions API 开放（[Google AI for Developers: Gemini Deep Research](https://ai.google.dev/gemini-api/docs/deep-research)、[tokencost.app breakdown](https://tokencost.app/blog/gemini-deep-research-agent-cost)）。Interactions API 于 2026-06-22 GA（[Interactions API GA](https://ai-watch-blog.vercel.app/en/posts/2026-06-22-gemini-interactions-api-ga)）。
- **架构**（来自 Google 自家文档与报道）：三段式流水线——**Planner（规划器）→ Browser（浏览器）→ Synthesizer（综合器）**。Planner 把问题拆成多步研究计划；Browser 执行 Google 搜索，并在允许时通过 MCP 跟随链接进入 Workspace 数据源（Gmail、Drive、Chat、BigQuery）；Synthesizer 编写 15+ 页带引用的报告，期间会做多次自我批评（[blog.google: Build with Gemini Deep Research](https://blog.google/technology/developers/deep-research-agent-gemini-api/)）。
- **运行时长**：最多约 60 分钟的自主执行。必须以异步方式调用（`background=true`）。
- **模型与定价**（[tokencost.app](https://tokencost.app/blog/gemini-deep-research-agent-cost)、[nowosci.ai](https://www.nowosci.ai/en/article/google-deep-research-max-api-agent)）：
  - `deep-research-preview-04-2026`——约 80 次查询，$1–3 / 任务
  - `deep-research-max-preview-04-2026`——约 160 次查询，$3–7 / 任务
  - 加 Google Search grounding：$14 / 1,000 次查询（默认启用）
  - 基础 Gemini 3 Pro：$2 / $12 每百万 token（≤200k 上下文）；缓存输入 $0.20 / 1M
- **基准**：HLE 46.4%、BrowseComp 59.2%、DeepSearchQA 66.1%。相比基线 Gemini，幻觉下降约 40%。
- **附带开源**：Google 开源了 **DeepSearchQA**——一个跨 17 个领域、900 个因果链任务的开源基准。

### 1.4 Anthropic — 带网页工具的 Claude

截至 2026-09-22，**Anthropic 没有"Deep Research"品牌产品**。他们发布的是**可由你自己组合进深度研究工作流的 server-side 工具**：

- **`web_search` 工具**——2025-05-07 上线（[anthropic.com/news/web-search-api](https://www.anthropic.com/news/web-search-api)）。三个版本：`web_search_20250305`、`web_search_20260209`（新增通过代码执行实现的动态过滤）、`web_search_20260318`（新增 `response_inclusion`，面向 agentic workflow）。定价 **$10 / 1,000 次搜索** 加 token 费用（[platform.claude.com: web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)）。
- **`web_fetch` 工具**——2025-09-10 上线。从一个 URL 返回全文内容（以及 base64 编码的 PDF）。引用为可选（`citations.enabled: true`）。版本演进：动态过滤、`use_cache` 绕过、`response_inclusion`。**除 token 外不另收费**（[platform.claude.com: web fetch tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool)）。
- **组合行为**：当两个工具都启用，且用户在消息中提及一个不带 URL 的页面时，Claude 会先搜索，再抓取。2026 版本的动态过滤让 Claude 能写代码、运行代码来过滤搜索 / 抓取结果，**在它们进入上下文之前**就把无关页面挡掉，从而显著降低检索密集型研究的 token 成本。
- **值得知道的限制**：`web_fetch` 不能渲染重 JS 页面（[platform.claude.com: web fetch tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool) 明确指引这类场景用 browser-use 工具）。`web_fetch` 还强制 URL 校验——Claude 只能抓取**已在会话其他地方出现过的 URL**（用户消息、之前的工具结果），而不能抓取只出现在 Claude 自己输出中的 URL，这能限制数据外泄与提示词注入风险。
- **对构建者的实际意义**：基于 Claude 的 deep research 产品本质上就是**你自己写的 agent loop**，把 Claude 的网页工具当作原语。Anthropic 还提供 **Managed Agents**（[platform.claude.com: Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview)），可用于托管一个长跑 agent，并对每个工具施加域名级别的限制。

### 1.5 Hugging Face Open Deep Research（smolagents）

- **是什么**：不是产品，而是 OpenAI Deep Research 的**开源复现**——由 5 位 HF 工程师在 2025 年 2 月用 24 小时冲刺完成，地址 [`huggingface/smolagents/examples/open_deep_research/`](https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research)。
- **架构（多 agent）**（[HF blog: Open Deep Research](https://huggingface.co/blog/open-deep-research)、README）：
  - **Manager agent**——一个 `CodeAgent`（动作以 Python 写出，而非 JSON），最多 12 步，每 N 步规划一次。
  - **Search sub-agent**——一个名为 `search_agent` 的 `ToolCallingAgent`，最多 20 步，每 4 步规划一次。Manager 向它委派任务。
  - **为什么用代码形式动作**：根据 [Wang et al. 2024 (arXiv:2402.01030)](https://arxiv.org/abs/2402.01030)，CodeAgent 比 JSON 工具调用 agent **少约 30% 的步数**，因为并行 / 顺序动作会折叠进同一段代码。状态（例如下载的图片）可以存在变量里。
- **内置工具**（大多借鉴自 Microsoft Magentic-One）：
  - `web_search`——默认使用 SerpApi（Google SERP）；可配置为 Serper.dev。
  - `visit_page`——纯文本浏览器，用 markdownify 去掉脚本 / 样式；视口 5120 字符。
  - `download_file`、`find_archived_url`（Wayback）、`PageUp/PageDown`、`find_on_page_ctrl_f`。
  - `inspect_file_as_text`——HTML / PDF / DOCX / XLSX / PPTX / 音频（通过 speech_recognition）/ 图片（OCR + 视觉描述）的文本检视器。
- **引用形式**：隐式。没有结构化的引用跟踪器。reformulation 步骤会强制一个严格的 `FINAL ANSWER:` 格式（为 GAIA 基准设计），行内引用则完全取决于 LLM 是否愿意把源 URL 写进去。
- **基准**：GAIA validation 55.15%，对比 OpenAI Deep Research 的 67.36%，以及纯 smolagents 基线约 33%。

### 1.6 其他 2025–2026 入场者

- **xAI Grok DeepSearch**——2025-02 随 Grok 3 上线（[x.ai/blog/grok-3-beta](https://x.ai/blog/grok-3-beta)）。架构是 Grok 3 之上的工具增强 agent：query planner → 并行 web / X retriever → evidence aggregator → Grok 3 "Think" 模式综合出带引用的答案。迭代地优化查询。SuperGrok 与 X Premium+ 订阅用户可通过 grok.com 与 X App 访问。
- **You.com**——You.com 自 2024 年起已发布 agent 式研究功能（Research agent、ARI），构建在自家"搜索 + 答案"混合栈之上（[you.com](https://you.com)）。
- **Mistral Le Chat Deep Research**——Mistral 在 Le Chat 中发布了"deep research"模式；底层模型是 Mistral 自家系列（Mixtral / Mistral Large）（[mistral.ai](https://mistral.ai)）。
- **DeepSeek DeepResearch**——DeepSeek 在其聊天产品中发布了 Deep Research agent（DeepSeek-V3.1 / R1 系列）（[deepseek.com](https://deepseek.com)）。以"开源权重骨干 + 激进定价"闻名。
- **Genspark**——产出带引用答案页的 AI 搜索引擎；2025 年逐步转向 agent 式多页报告。
- **Felo / Skywork / Manus / Genspark 等**——2025 年长尾入场者，都是"agent + 大量搜索 + 带引用报告"的变体。截至 2026 年中，没有任何一家撼动顶部三强。

---

## 2. 共同架构模式

每一个真正上线的深度研究系统，骨子里都是同一件事：**一个推理模型在一个工具调用循环里**。下面给出这个内部循环的"标准形态"，并标注谁做了什么。

### 2.1 查询分解 / 规划

- **OpenAI Deep Research**——隐式、在链中。System Card 把该模型描述为"把用户任务分解为子问题，并决定下一步该搜什么"（[System Card](https://cdn.openai.com/deep-research-system-card.pdf)）。它还会就期望的范围、时间范围、来源偏好向用户提问澄清。
- **Gemini Deep Research**——显式的 Planner。Google 文档把 **Planner** 描述为一个独立阶段，产出多步研究计划，并在运行前展示给用户做审核 / 修改（[blog.google](https://blog.google/technology/developers/deep-research-agent-gemini-api/)）。
- **Hugging Face Open Deep Research**——Manager `CodeAgent` 用 Python 写出研究轨迹，再按子任务委派给 search sub-agent（[README](https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research)）。
- **smolagents 通用做法**——每 N 步规划一次（可配），把下一次"动作"的规划写成 Python 注释块。

### 2.2 搜索查询生成（数量、迭代深度）

- **OpenAI**——每次运行数十到约数百次查询。模型会基于学到的东西**优化查询**，并**能回溯**（[System Card](https://cdn.openai.com/deep-research-system-card.pdf)）。
- **Perplexity**——监督者针对同一个子问题"并行做很多事"，再去重（[Perplexity Sonar architecture FAQ](https://www.perplexity.ai/hub/faq/explain-the-sonar-architecture)）。
- **Gemini**——同样采用并行搜索；Google 文档把 Browser 阶段描述为：搜索 → 解析 → 识别缺口 → 再次搜索 的迭代。
- **Open Deep Research**——search sub-agent 有硬上限 20 步（也就是最多 20 次搜索 / 浏览动作）。

### 2.3 网页抓取 vs 搜索 API 模式

本质上两种模式：

1. **搜索 API → URL → 抓取**。搜索返回排过序的 URL 列表；agent 再抓取完整页面（HTTP GET，重 JS 站点可能要无头浏览器）。绝大多数生产系统走这条路。每一层的具体工具可配置（见 §3）。
2. **原生 server-side 网页工具**。Anthropic 的 `web_search` + `web_fetch`，以及 OpenAI Responses API 的 `web_search` 工具，把两层折叠成一个由 provider 托管的调用。代价是控制力变弱，工程量也变少。

### 2.4 来源排序、过滤、去重

- **Open Deep Research** 没有形式化去重——完全依赖 LLM 自身。
- **Perplexity** 明确运行并行检索尝试，并在综合前按内容哈希去重。
- **Anthropic 的动态过滤**（2026 版本）是一个显著的新模式：Claude 不是把每个搜索结果都倒进上下文，而是**写并执行一段过滤代码，把无关页面在进入 token 预算之前挡掉**（[web search tool docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)）。相比朴素的检索，这能在多步研究中显著降低成本。

### 2.5 阅读与笔记

每个系统都会对抓取到的页面做摘要，但摘要的"存法"各有不同：

- **内联到上下文**（OpenAI Deep Research、Anthropic Claude、Open Deep Research）——agent 的对话历史**本身就是笔记本**。
- **显式 scratchpad**——Gemini 的 Planner 维护一个结构化计划，改写后仍能保留。
- **落盘 / 沙箱文件系统**——Open Deep Research 让 manager 下载文件（`.xlsx`、`.pdf`、`.png`），再用 `inspect_file_as_text` 重新检视；这把多模态输入变成"通过工具处理"，而不是把字节塞进 prompt。

### 2.6 综合（Sythesis）

最终还会跑一次"写作"步骤，这是普遍做法：

- **OpenAI**——模型本身在同一个 loop 里生成最终报告；之后**再**调用一次 o3-mini，把 chain-of-thought 摘要给侧边栏（[System Card](https://cdn.openai.com/deep-research-system-card.pdf)）。
- **Gemini**——Synthesizer 阶段显式做多次自我批评（self-critique）。
- **Open Deep Research**——`reformulator.py` 后处理步骤向模型喂"原始任务 + 完整 transcript"，强制它产出一个 `FINAL ANSWER: ...` 抽取，最初是为 GAIA 风格的精确答案打分设计的。

### 2.7 引用处理

这是各家分歧最大的一块：

- **行内编号引用**（OpenAI、Perplexity、Gemini）——用户体验最干净。
- **带字符偏移的结构化引用**（Anthropic `web_fetch` 的 `char_location` 块）——最严谨；每条引用都带 `cited_text` 原文片段与精确字符跨度。
- **隐式、机会主义式**（Open Deep Research）——由模型自己决定要不要提到来源。生产环境不可靠。

### 2.8 迭代深度与停止条件

- **OpenAI**——一直跑到模型输出 final answer 为止；深度受时间和步数约束。
- **Gemini**——使用一个"async task manager"在 Planner 与任务模型之间维护共享状态，支持优雅恢复和带步数上限的循环（[Google blog](https://blog.google/technology/developers/deep-research-agent-gemini-api/)）。
- **Open Deep Research**——硬上限：manager 12 步，search sub-agent 20 步。
- **Anthropic**——可对每个工具设置 `max_uses`（例如 `web_fetch: max_uses: 10`）——既是硬上限，也是一种成本控制。

### 2.9 多 agent vs 单 agent

- **带规划能力的单 agent** 是默认：OpenAI Deep Research、带网页工具的 Claude、Perplexity Deep Research、Gemini Deep Research（Planner / Browser / Synthesizer 都是同一个编排内的阶段）。
- **层级式多 agent** 较少见但存在：Open Deep Research 的 manager + search-agent 分工。LangChain 自家的 `open_deep_research` 在历史上反复在多 agent 与单 loop 之间摇摆，值得一提的是——**单 loop 版本现在在 DeepResearch Bench 上击败了多 agent 版本**（[dreaming.press overview](https://dreaming.press/posts/open-source-deep-research-agents.html)）。
- **2026 年浮现的通用规律**：当底层模型足够强、能自己规划时，多 agent 的边际收益递减；多 agent 仍然有用的，是当各 agent 需要不同工具或不同 system prompt 时（例如一个写代码、一个搜网页）。

---

## 3. 积木——API 与工具

价格均为撰写时（2026 年中）各家官方页面所列，预算前请再确认一次。

### 3.1 搜索 API

| 厂商 | 是什么 | 价格（2026 年中） | 备注 |
|---|---|---|---|
| **Tavily** ([tavily.com](https://tavily.com)) | 一站式搜索 + 抽取，专为 agent 设计 | 免费：1,000 积分/月。Project $30/4k，Startup $220/38k，Growth $500/100k。基础搜索 = 1 积分，高级 = 2 积分。 | 自带内容抽取（省一次往返）。SimpleQA 93.3%。([docs.tavily.com](https://docs.tavily.com)) |
| **Exa** ([exa.ai](https://exa.ai)) | 基于自家索引的神经 / 语义搜索 | 注册赠 $20 + 每月 $10 免费。Starter $49，Pro $449。按请求：神经 1–25 结果 $5/1k，神经 26–100 $25/1k；**Exa Deep $12–15/1k**。 | 自称索引约 1,000 亿文档；Exa Deep 跑多搜索 + 排序（[exa.ai/changelog](https://exa.ai/docs/changelog)）。 |
| **Brave Search API** ([brave.com/search/api](https://brave.com/search/api/)) | 独立网页索引 | 免费：2,000 查询/月 @ 1 QPS。Pro AI：$3–5/CPM；Pro Data：$5–9/CPM。 | Data 档位没有 AI 再加工；AI 档位带摘要。 |
| **Serper** ([serper.dev](https://serper.dev)) | Google SERP API | 免费：2,500 查询（一次性）。Starter $50/50k，Growth $130/200k。规模化时约 $0.30/1k。 | 便宜、快、Google 形态的结果。HF Open Deep Research 的默认。 |
| **Google Programmable Search Engine (CSE)** ([developers.google.com/custom-search](https://developers.google.com/custom-search)) | 官方 Google 站点限定搜索 | 前 100/天免费，超出 $5/1k（上限 10k/天）。 | 限制多、价格贵；只有合规上必须用 Google 索引时才有意义。 |
| **Bing Web Search API** ([learn.microsoft.com/bing/search](https://learn.microsoft.com/en-us/bing/apis/bing-web-search-api)) | Microsoft Bing 索引 | Azure 上免费 1k/月。S1 $3/1k（最多 250k/月），S2 $7/1k，S3 定制。 | 微软栈客户的标准选择。 |
| **Perplexity Sonar** | 已在 §1.2 覆盖 | $5/1k 搜索（与 Exa 神经版一致） | 搜索 API 之上的"答案引擎"层。 |

**战术建议**：MVP 阶段先上 **Tavily**（便宜、自带抽取）或 **Exa**（神经质量）。在后台为高量 SERP 数据再加一层 **Serper**。把 **Brave** 作为隐私友好的第二索引备用。

### 3.2 网页抓取与爬取

| 厂商 | 是什么 | 价格 | 备注 |
|---|---|---|---|
| **Firecrawl** ([firecrawl.dev](https://firecrawl.dev)) | 托管抓取 + markdown 转换 + 爬取 | 免费：500 积分。Hobby ~$19/3k，Standard ~$99/25k，Pro ~$299/100k，Enterprise 定制（[firecrawl.dev/pricing](https://firecrawl.dev/pricing)）。 | 五类操作：Scrape、Crawl、Extract（LLM 驱动的 JSON）、Map（URL 发现）、Search。JS 渲染、隐身代理。 |
| **Jina Reader** ([jina.ai/reader](https://jina.ai/reader)) | `r.jina.ai/{URL}` 返回 LLM 友好的 markdown；`s.jina.ai/{query}` 做搜索 | 免费，200 RPM，无需 API key。 | 通过 Header 控制行为：`X-Target-Selector`、`X-Wait-For-Selector`、`X-Respond-With`。支持 PDF。免费档位最强。 |
| **Playwright** ([playwright.dev](https://playwright.dev)) | 开源浏览器自动化 | 免费；自己跑浏览器 | 处理重 JS 站点与已登录流程的默认选择。 |
| **Browserbase** ([browserbase.com](https://browserbase.com)) | 托管 Playwright / Puppeteer / Selenium | 免费 ~1 小时。Developer ~$20/10 小时。Production 定制。 | 隐身模式、验证码求解、100+ 国家。与 LangChain、CrewAI 有对接。 |
| **Browserless, Steel.dev, Anchor Browser, Hyperbrowser** | 备选 | 各异 | Browserbase 当前在 agent 生态里被采用最多。 |
| **Anthropic `web_fetch`** | 由 provider 托管 | 不另收费 | 仅限文本 / HTML / PDF，无 JS。URL 校验防止数据外泄。 |
| **OpenAI `web_search` 工具**（Responses API） | 由 provider 托管 | 见 OpenAI 定价 | 打包在 Responses API 内。 |

**战术建议**：典型的 deep research 栈会用 **Jina Reader** 处理 80% 的简单页面，用 **Firecrawl 或 Browserbase** 处理长尾的 JS 渲染或反爬页面。Anthropic 的 `web_fetch` 适合"已经过滤到高价值 URL 之后"再用（它不跑 JS）。

### 3.3 带原生网页工具的 LLM API

| 提供方 | 工具 | 价格 | 亮点 |
|---|---|---|---|
| **Anthropic Claude** | `web_search_20260318`、`web_fetch_20260318` | web_search：$10/1k；web_fetch：免费（只算 token） | 通过代码执行实现动态过滤；URL 校验；`response_inclusion` 省 token（[docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)）。 |
| **OpenAI** | Responses API `web_search` 工具 | 按 OpenAI 公开的工具定价 | 集成在 Chat Completions / Responses 中；撰写本文时尚未提供独立的 Deep Research API。 |
| **Google Gemini** | Google Search grounding | $14 / 1,000 次 grounding 查询 | 在 Deep Research 上默认开启（[ai.google.dev/gemini-api/docs/deep-research](https://ai.google.dev/gemini-api/docs/deep-research)）。 |

### 3.4 Agent 框架

- **LangGraph** ([github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph))——基于图、状态化的 agent loop。在确定性、长跑、有人参与的研究里胜出，支持 checkpoint。官方有一个 **Open Deep Research** 模板（`langchain-ai/open_deep_research`），在 DeepResearch Bench 上对比后**主动把旧版 supervisor-researcher 多 agent 方案替换成了单 loop 版本**。
- **smolagents** ([github.com/huggingface/smolagents](https://github.com/huggingface/smolagents))——`CodeAgent` 把动作写成 Python，比 JSON agent **少约 30% 步数**。核心代码约 1,000 行。自带 Open Deep Research 示例。沙箱选项：E2B、Blaxel、Modal、Docker（LocalPythonExecutor **明确不充当安全边界**）。
- **CrewAI** ([github.com/crewAIInc/crewAI](https://github.com/crewAIInc/crewAI))——基于角色的"crew"式 agent。适合快速原型：每个 agent 角色清晰（Researcher、Writer、Critic）。
- **AutoGen** ([github.com/microsoft/autogen](https://github.com/microsoft/autogen))——Microsoft Research。可对话 agent + 代码执行；v0.4+（2024–2025）引入了异步事件驱动架构。Magentic-One（HF Open Deep Research 倚重的多 agent 研究编排器）也属于这一系。
- **Pydantic AI** ([github.com/pydantic/pydantic-ai](https://github.com/pydantic/pydantic-ai))——类型安全、Pydantic 原生。在需要结构化输出与校验时很强。
- **OpenAI Agents SDK** ([github.com/openai/openai-agents-python](https://github.com/openai/openai-agents-python))——官方轻量 SDK，针对 OpenAI 工具调用模型；带 handoff、追踪、会话管理。
- **Google ADK** ([github.com/google/adk-python](https://github.com/google/adk-python))——Agent Development Kit，2025 发布；与 Interactions API 对齐。

### 3.5 RAG / 向量组件

标准 pgvector / Qdrant / Weaviate 作为**抓取后存储**很有用——当你希望恢复一个长跑研究会话、跨用户共享语料、或缓存结果时。主要的托管 deep research 产品默认都不暴露向量库——它们在整个运行期间把一切留在模型上下文里。如果要做跨会话的研究，规划上加一个带元数据的向量存储（URL、时间戳、产出该片段的查询），让后续运行可以短路。

---

## 4. 给构建者的具体架构选择

以下是跨产品与跨框架正在浮现为共识的模式。证据支持的地方，给出明确的观点。

### 4.1 串行 vs 并行 研究子 agent

**并行的优势场景**：(a) 子问题之间相互独立；(b) 底层 LLM 强到能去重；(c) 延迟敏感。Perplexity 对同一个子问题跑大量并行尝试，然后按内容哈希去重。Gemini 的 Browser 阶段对多个页面并行抓取。

**串行的优势场景**：下一个问题依赖上一个答案时（例如"头条结果里关于 X 说了什么？"）。纯并行用在一组有依赖的问题上只会产生大量重复抓取。

**v1 实用默认**：一个 manager 跑紧凑的串行 loop，在**每一步内部做并行抓取**（一次搜索 → N 个并行的 `web_fetch` → 一步综合）。只有当单 loop 出现明显瓶颈后，再升级到多 agent。

### 4.2 单遍 vs 迭代加深

单遍（"一次生成全部查询、抓完全部页面、写报告"）快但浅。迭代加深（"读、识别缺口、再搜"）能抓到漏掉的来源与矛盾，但成本成倍上升。

- **OpenAI** 与 **Gemini** 都是强迭代（5–30 分钟 / 最多 60 分钟）。
- **Open Deep Research** 用硬步数上限（manager 12、search 20）强制有界深度。
- **Anthropic 的 `max_uses`** 让你能对 `web_search` 与 `web_fetch` 分别设上限——是约束深度预算的干净手段。

**实用默认**：迭代 loop + 硬上限。使用**提示词层的停止准则**（"如果你对一条论断已有 ≥3 个独立来源确认，就继续推进"）以及**预算级断路器**（最大步数、最大墙钟时间、每次运行最大美元数）。

### 4.3 引用落地（Citation grounding）

这是单一最大的质量杠杆。三种值得叠加的模式：

1. **来自搜索 / 抓取工具的原生引用**。Anthropic 的 `web_fetch` 返回带 `cited_text` 原文片段与精确字符跨度的 `char_location` 块；Gemini 返回带来源归属的 grounding chunks；OpenAI Deep Research 暴露一个 sources 面板。用它们——别再靠"URL 字符串回贴到 LLM 输出"重建引用。
2. **在写作时绑定引用，而不是写完再插**。Gemini 的 Synthesizer 阶段就明确这么做；引用是写作步骤的一部分，所以能跨改写存活。如果你后期处理再加引用，就会出现"声明和引用不一致"。
3. **可校验的原文片段**。当你掌控流水线时（例如 Tavily 或 Firecrawl + 你自己的 LLM），把声明引用的那段实际文本片段也带进引用元数据里。这让事后校验变得简单。

### 4.4 控制成本与延迟

真实世界的成本形态：

- **OpenAI Deep Research** 每次报告跑下来消耗数十次 o3 调用 + 数十次网页抓取。折算成推理成本约 $5–50 / 报告，取决于深度。
- **Gemini Deep Research API** $1–7 / 任务，加 $14 / 1k grounding 查询（[tokencost.app](https://tokencost.app/blog/gemini-deep-research-agent-cost)）。
- **Anthropic** $10 / 1k 次搜索 + token；`web_fetch` 免费但 token 加起来也不小。

工程上的控制手段：

- **每个 agent 的硬步数上限**（Open Deep Research 风格）。
- **按工具的 `max_uses`**（Anthropic 风格）。
- **每次运行的墙钟与美元预算**，**提前**告知用户（"本次最多跑 X 分钟、$Y"）。
- **对静态工具定义与重复研究计划做 prompt 缓存**（Anthropic 支持；OpenAI、Google 有等价物）。
- **动态过滤**（Anthropic 2026 版本）把无关搜索 / 抓取挡在 token 账外。
- **边写边流式输出报告**，而不是只在最后才出——这是所有已上线产品的标配。

### 4.5 评估

要追踪的基准与 harness：

- **GAIA**——多模态、需要工具的 agent 基准。"deep research" 类 agent 的头条数字（[Hugging Face: GAIA-annotated](https://huggingface.co/datasets/smolagents/GAIA-annotated)）。
- **Humanity's Last Exam (HLE)**——推理基准。OpenAI Deep Research 26.6%；Gemini Deep Research 46.4%。
- **BrowseComp**——浏览专用基准。Gemini Deep Research 59.2%。
- **DeepSearchQA**——Google 随 Deep Research agent 一同开源：跨 17 个领域、900 个因果链任务（[blog.google](https://blog.google/technology/developers/deep-research-agent-gemini-api/)）。
- **DeepResearchBench**——LangChain `open_deep_research` 用来证明"单 loop 设计击败自家旧版多 agent"的基准。

对内部评估，搭一个小集合：(a) 有已知好答案的问题；(b) 故意设计的引用陷阱题。衡量：

1. **引用有效性**（URL 能否解析到一个真实页面？）
2. **蕴含性**（被引用的页面是否真的支持该论断？）
3. **覆盖度**（决策性论断是否有 ≥2 个独立来源？）
4. **截断下的忠实度**（运行中途截断上下文后，报告是否仍能存活？）

---

## 5. 风险与坑

关于 deep-research-agent 失败的公开文献，比泛泛的"LLM 幻觉"建议有用得多。主要参考：[arXiv 2604.03173](https://arxiv-vanity.com/papers/2604.03173)（Reference Hallucinations in Commercial LLMs and Deep Research Agents）、[arXiv 2608.28643](https://arxiv.org/html/2608.28643)（Faithful Reports）、[LinkedIn: Up to Half of Your AI Research Assistant's Citations Are Fake](https://www.linkedin.com/pulse/up-half-your-ai-research-assistants-citations-fake-real-bilal-nm7uf)，以及 [fieldway.org](https://fieldway.org/blog/research-agents-get-less-accurate-the-more-they-read-42-worse-from-2-tool-calls-to-150) 关于"工具调用越多、准确率越差"的分析。

### 5.1 幻觉来源与引用漂移（citation drift）

- **3–13% 的引用 URL 是幻觉**；**5–18% 无法解析**。Deep research agent 引用最多，但**伪造率也最高**。
- **声明级错引**：即便 URL 是真的，OpenAI Deep Research 约 50% 的被引声明仍含细微不准确；Gemini 2.5 Pro 与 Perplexity Deep Research 在某些领域约 47–50% 的引用是伪造的。
- **随工具调用深度的引用漂移**：当工具调用从 2 次增长到 150 次时，事实核查准确率下降约 42%。链接有效性保持较高（>92%）；退化出在**综合环节**，而非来源选择环节。GPT-5.4：79% → 17%；Claude Opus 4.6：80% → 58%。
- **多轮漂移**：29–86% 的引用在会话轮次之间会发生变化。

**缓解措施**：写作时绑定引用；在引用元数据里带上被引用的原文片段；永远不要让模型引用它并未实际抓取过的 URL；运行结束后服务端校验 URL。

### 5.2 付费墙与动态内容

- 大多数搜索 API 只返回可被爬取的内容；付费墙内容（NYT、学术期刊）看不见。
- JS 渲染的页面（SPAs）通常需要无头浏览器。Anthropic 的 `web_fetch` 明确不渲染 JS——这类场景需要 Playwright 或 Browserbase 配合。
- 实时价格、即时比分、快速变化的页面可能需要 `use_cache: false`（Anthropic 2026 版本）或 Brave 的时效控制。

### 5.3 速率限制与成本失控

- 长跑研究会悄悄冲破速率限制——一个 30 分钟的运行做 200 次网页抓取就会超过绝大多数厂商的每秒上限。
- 做一个**预算守护**：跟踪美元数与工具调用数，超限时优雅中止，并清晰告诉用户"我停下来是因为预算用完了"。
- Tavily 开发档位上限 100 RPM；生产 1,000 RPM。Exa Starter 慷慨，但 Exa Deep $12–15/1k 累计起来也快。Anthropic $10/1k 次搜索意味着 100 次搜索就要 $1，外加 token。

### 5.4 反爬保护

Cloudflare、DataDome、Akamai 会拦截朴素爬取。Firecrawl 的隐身模式、Browserbase 的隐身模式、住宅代理网络能帮上忙，但都不是免费的，也都不是万能的。设计优雅失败：如果页面返回 403 或"请验证你是人类"，agent 应回退到：(a) 缓存 / 归档版本（Wayback Machine——Open Deep Research 中的 `find_archived_url`）；(b) 替代来源；(c) 直接承认它访问不了该页。

### 5.5 来自网页内容的提示词注入

OpenAI 的 System Card 明确把这列为头号风险。Anthropic 的 `web_fetch` 强制 URL 校验（Claude 只能抓取在更早对话上下文中出现过的 URL），正是为了限制"通过抓取完成的注入"。防御手段：沙箱化代码执行；把抓取到的文本当作不可信输入；在被抓取内容接触特权工具调用前，必须经过一次模型侧的判断。[DeepResearchGuard (ACL 2026)](https://hyper.ai/en/papers/acl__2026__2026.acl-long.2010) 是目前最完整的公开方案，提供了四阶段的安全防护流水线。

### 5.6 长程漂移（Long-horizon drift）

如上所述，agent 跑得越久，综合质量越差——即便来源选择依然准确。反直觉的是，**单遍报告可能比 30 分钟的研究马拉松更忠实**。实际含义：**别追求更多步骤，要追求更少、更好的步骤**。能并行的就并行；限定深度；边写边把引用钉死。

### 5.7 评估盲点

已发布的多数基准（GAIA、HLE、BrowseComp）衡量的是"agent 是否答对"，而不是"agent 是否准确且忠实地引用"。生产环境里，你需要一份"引用忠实度"评估，与"答案正确性"评估并列。Salesforce 出品的 [ClaimProbe / ClaimWriter](https://arxiv.org/abs/...) 与 [DeepTRACE](https://liner.com/review/deeptrace-auditing-deep-research-ai-systems-for-tracking-reliability) 是当前公开 harness 里较领先的两个。

---

## 来源索引

**官方产品页与 System Card**
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
- LangChain: [open_deep_research](https://github.com/langchain-ai/langgraph)（见 `examples/open_deep_research`）

**API 厂商文档**
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

**评估与风险文献**
- arXiv: [Executable Code Actions Elicit Better LLM Agents (Wang et al., 2024)](https://arxiv.org/abs/2402.01030)
- arXiv: [Detecting and Correcting Reference Hallucinations](https://arxiv-vanity.com/papers/2604.03173)
- arXiv: [Faithful Reports (Redesigning Deep Research Writing)](https://arxiv.org/html/2608.28643)
- ACL 2026: [DeepResearchGuard](https://hyper.ai/en/papers/acl__2026__2026.acl-long.2010)
- fieldway.org: [Research Agents Get Less Accurate the More They Read](https://fieldway.org/blog/research-agents-get-less-accurate-the-more-they-read-42-worse-from-2-tool-calls-to-150)
- LinkedIn: [Up to Half of Your AI Research Assistant's Citations Are Fake](https://www.linkedin.com/pulse/up-half-your-ai-research-assistants-citations-fake-real-bilal-nm7uf)
- DeepTRACE: [Liner summary](https://liner.com/review/deeptrace-auditing-deep-research-ai-systems-for-tracking-reliability)

**定价与覆盖参考**
- tokencost.app: [Gemini Deep Research cost breakdown](https://tokencost.app/blog/gemini-deep-research-agent-cost)
- nowosci.ai: [Google Deep Research Max](https://www.nowosci.ai/en/article/google-deep-research-max-api-agent)
- TechCrunch: [Gemini can now research deeper (Dec 2024)](https://techcrunch.com/2024/12/11/gemini-can-now-research-deeper/)
- dreaming.press: [Open-Source Deep Research Agents overview](https://dreaming.press/posts/open-source-deep-research-agents.html)