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

QUERY = "Python 3.13 release notes"
TOPIC_ID = "topic-1"

raw = get_web_search().invoke({"query": QUERY, "topic_id": TOPIC_ID})
rprint(raw)

# Validate through the same Pydantic model the production code uses.
parsed = SearchResult(**raw)
assert parsed.url, f"search returned empty url: {raw!r}"
assert parsed.title, f"search returned empty title: {raw!r}"
print(f"\nOK: {parsed.title} — {parsed.url}")
