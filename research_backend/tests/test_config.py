"""Config loading — ``.env`` is read on first get_config(), but env wins.

The autouse ``_reset_config_cache`` fixture in conftest.py noops
``load_dotenv`` for every test; here we re-enable it with a call-tracker
to verify the production contract without ever touching the real ``.env``
file (which ``find_dotenv`` would always locate under the project root).
"""

from __future__ import annotations

import pytest

from app import config as config_module
from app.config import get_config


@pytest.fixture
def track_load_dotenv(monkeypatch: pytest.MonkeyPatch):
    """Replace ``app.config.load_dotenv`` with a call-tracker.

    The tracker records every invocation and (optionally) installs fake
    values into ``os.environ`` to simulate what a real ``.env`` would do.
    """
    calls: list[dict] = []

    def _tracker(**kwargs):
        calls.append(kwargs)
        return None  # don't actually touch os.environ

    monkeypatch.setattr(config_module, "load_dotenv", _tracker)
    return calls


def test_load_dotenv_called_with_override_false(track_load_dotenv) -> None:
    """Production must call ``load_dotenv(override=False)`` so env wins."""
    get_config()
    assert track_load_dotenv == [{"override": False}]


def test_load_dotenv_runs_only_on_first_call(track_load_dotenv) -> None:
    """Subsequent ``get_config()`` calls reuse the cache and skip dotenv."""
    get_config()
    get_config()
    get_config()
    assert len(track_load_dotenv) == 1


def test_get_config_reflects_env_var(
    monkeypatch: pytest.MonkeyPatch, track_load_dotenv
) -> None:
    """Env vars set before ``get_config()`` are reflected in the Config."""
    monkeypatch.setenv("OPENAI_API_KEY", "from-os-environ")
    cfg = get_config()
    assert cfg.openai_api_key == "from-os-environ"


def test_get_config_falls_back_to_default_when_env_unset(
    monkeypatch: pytest.MonkeyPatch, track_load_dotenv
) -> None:
    """Env vars unset → Config uses the dataclass default (NOT a dotenv value)."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    cfg = get_config()
    assert cfg.openai_base_url == "https://api.openai.com/v1"
