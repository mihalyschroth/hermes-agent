"""Regression tests for the Codex 401 → import-from-Codex-CLI recovery path.

When Hermes's own Codex refresh token is rotated (and thereby invalidated) by
another client — Codex CLI or the VS Code extension — its own refresh either
fails outright or yields a token the backend still rejects with HTTP 401
``token_expired``. The runtime diagnostic tells the user to run ``codex`` then
``hermes auth`` to re-import the live tokens; ``_try_import_codex_cli_credentials``
automates that re-import as a last-resort recovery before giving up.
"""

from unittest.mock import MagicMock


def _make_codex_agent():
    """Build a minimal AIAgent wired for openai-codex codex_responses."""
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="stale-token",
        base_url="https://chatgpt.com/backend-api/codex",
        model="gpt-5.5",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "codex_responses"
    agent.provider = "openai-codex"
    agent._interrupt_requested = False
    return agent


def test_import_recovery_adopts_fresh_token_and_rebuilds_client(monkeypatch):
    """A fresh CLI token replaces the stale one and rebuilds the client."""
    agent = _make_codex_agent()
    agent._replace_primary_openai_client = MagicMock(return_value=True)

    monkeypatch.setattr(
        "hermes_cli.auth.import_codex_cli_runtime_credentials",
        lambda: {
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "fresh-token",
            "source": "codex-cli-import",
        },
    )

    assert agent._try_import_codex_cli_credentials() is True
    assert agent.api_key == "fresh-token"
    assert agent._client_kwargs["api_key"] == "fresh-token"
    agent._replace_primary_openai_client.assert_called_once()
    assert (
        agent._replace_primary_openai_client.call_args.kwargs["reason"]
        == "codex_cli_token_import"
    )


def test_import_recovery_returns_false_when_nothing_to_import(monkeypatch):
    """No usable CLI token → no retry."""
    agent = _make_codex_agent()
    agent._replace_primary_openai_client = MagicMock(return_value=True)
    monkeypatch.setattr(
        "hermes_cli.auth.import_codex_cli_runtime_credentials",
        lambda: None,
    )

    assert agent._try_import_codex_cli_credentials() is False
    agent._replace_primary_openai_client.assert_not_called()


def test_import_recovery_returns_false_for_same_token(monkeypatch):
    """Re-importing the token we already failed with must NOT burn a retry."""
    agent = _make_codex_agent()  # api_key="stale-token"
    agent._replace_primary_openai_client = MagicMock(return_value=True)
    monkeypatch.setattr(
        "hermes_cli.auth.import_codex_cli_runtime_credentials",
        lambda: {
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "stale-token",
        },
    )

    assert agent._try_import_codex_cli_credentials() is False
    agent._replace_primary_openai_client.assert_not_called()


def test_import_recovery_skips_xai_oauth(monkeypatch):
    """xai-oauth doesn't share ~/.codex — the import path must not fire."""
    agent = _make_codex_agent()
    agent.provider = "xai-oauth"
    sentinel = {"called": False}

    def _should_not_run():
        sentinel["called"] = True
        return {"api_key": "x", "base_url": "y"}

    monkeypatch.setattr(
        "hermes_cli.auth.import_codex_cli_runtime_credentials", _should_not_run
    )

    assert agent._try_import_codex_cli_credentials() is False
    assert sentinel["called"] is False


def test_import_recovery_returns_false_when_client_rebuild_fails(monkeypatch):
    """If the client can't be rebuilt, recovery reports failure."""
    agent = _make_codex_agent()
    agent._replace_primary_openai_client = MagicMock(return_value=False)
    monkeypatch.setattr(
        "hermes_cli.auth.import_codex_cli_runtime_credentials",
        lambda: {
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "fresh-token",
        },
    )

    assert agent._try_import_codex_cli_credentials() is False
