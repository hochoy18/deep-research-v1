# Deep Research Agent — 后端领域术语表

Deep Research 产品的后端：接收用户问题，规划结构化研究，收集网络证据，产出带引用的报告。本术语表定义后端使用的规范语言；跨切面的决策记录在 `docs/adr/`。

## 流水线

**Thread**:
针对单个用户问题的一次端到端流水线执行。由后端生成的 UUIDv4 标识。
_Avoid_: session、会话、conversation、request

**Plan stage**:
第一阶段。通过多轮结构化提问澄清用户意图，并把问题拆解为 Research 阶段将要执行的主题列表。
_Avoid_: clarify stage、intake

**Research stage**:
第二阶段。按主题列表发起网络搜索并收集证据，产出结构化的检索结果集合。
_Avoid_: search stage、gather、retrieve

**Write stage**:
第三阶段。基于已收集证据产出报告大纲，撰写报告草稿，按三个维度评审草稿，再润色。
_Avoid_: output stage、draft stage

**Transition gate**:
两个阶段之间的人机协作检查点，用户可在此审阅并编辑交接产物（主题列表），下一阶段才会启动。
_Avoid_: review gate、handoff

## 研究产物

**Topic**:
从用户整体问题中拆出的单个子问题，范围足够窄，可用一组有限的网络搜索来回答。research plan 即一组 Topic。
_Avoid_: subtopic、question、query

**Search result**:
一次检索到的一个网页：其 URL、标题、摘要，以及（可选的）提取全文。携带来自搜索服务的相关性评分。
_Avoid_: hit、document、source page

**Reference**:
最终报告里的引文，指向某个 Search result：正文以行内 `[N]` 标记，文末附编号列表。
_Avoid_: citation、source、link、footnote

## 写作产物

**Outline**:
报告的结构骨架：一组 H2 章节，每个章节包含标题、一句话摘要、要点列表。
_Avoid_: structure、TOC、headings

**Draft**:
由 Outline 产出的第一版散文，未经过评审或润色。可根据 Review 反馈迭代多轮。
_Avoid_: report、response、final answer

**Review**:
对 Draft 的批评，沿三个维度打分：事实准确性、结构清晰度、引用完整性。产出 review notes 驱动草稿修订。
_Avoid_: critique、audit、evaluation

**Polish**:
对已接受的 Draft 做最终润色：衔接更顺畅、术语与语气统一、修正错别字与标点。不改变事实、结构或论断。
_Avoid_: refine、edit、format

## 人机协作

**Interrupt**:
流水线中的暂停点，执行挂起并等待用户输入后再继续。后端将每个 Interrupt 暴露为类型化载荷，前端据此渲染。
_Avoid_: halt、stop、wait

## 质量门槛

**Three-dimensional review**:
每份 Draft 在进入润色前必须同时满足 Review 的三个维度：每条论断都有来源支撑、结构贴合 Outline、所有引用过的来源都出现在 References 列表中。详见 ADR-0007。
_Avoid_: holistic review、gut check