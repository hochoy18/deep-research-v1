"""Web search verification case — call the ``web_search`` tool end-to-end.

Exercises the same code path ``research_run`` uses (ticket #4 / ADR-0010):
a single ``dashscope.Application.call(app_id, prompt)`` against a Bailian
single-agent application configured with a web-search plugin. The app
returns structured JSON; the tool projects ``pages[0]`` into the canonical
``SearchResult``. Prints the result and asserts the two fields that matter
(``url``, ``title``) are non-empty so a quota / auth failure shows up as
a clear assertion error, not a silent empty dict.

Configuration comes from the same ``.env`` the rest of the project uses
(``DASHSCOPE_API_KEY`` + ``DASHSCOPE_MCP_APP_ID``). To target a real
Bailian app instead of a stub, edit ``.env``::

    DASHSCOPE_API_KEY=sk-your-real-dashscope-key
    DASHSCOPE_MCP_APP_ID=<your-bailian-app-id>
"""

from app.models import SearchResult
from app.tools.web_search import get_web_search
from rich import print as rprint

QUERY = "调研欧美国家 酒店行业 在AI Agent 的落地使用"
TOPIC_ID = "topic-1"

raw = get_web_search().invoke({"query": QUERY, "topic_id": TOPIC_ID})
rprint(raw)

# The tool returns one SearchResult per Bailian page (typically ~5).
# Validate through the same Pydantic model the production code uses.
parsed = [SearchResult(**p) for p in raw]
assert parsed, f"search returned no pages: {raw!r}"
for p in parsed:
    assert p.url, f"search returned empty url: {p!r}"
    assert p.title, f"search returned empty title: {p!r}"
print(f"\nOK: {len(parsed)} pages — {parsed[0].title} — {parsed[0].url}")
