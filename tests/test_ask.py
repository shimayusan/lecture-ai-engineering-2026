from types import SimpleNamespace

import pytest
from google.genai import types as genai_types

from app import ask as ask_module


def test_ask_passes_model_and_temperature_and_prints_metrics(monkeypatch, capsys):
    response = SimpleNamespace(
        text="回答",
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=34),
    )
    request = {}

    class Models:
        def generate_content(self, **kwargs):
            request.update(kwargs)
            return response

    class Client:
        models = Models()

    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.genai, "Client", lambda api_key: Client())

    assert ask_module.ask("質問", model="test-model", temperature=0.5) == "回答"

    assert request == {
        "model": "test-model",
        "contents": "質問",
        "config": genai_types.GenerateContentConfig(temperature=0.5),
    }
    output = capsys.readouterr().out
    assert "[応答時間]" in output
    assert "[トークン数] 入力: 12, 出力: 34" in output


def test_session_manager_keeps_history_until_session_ends(monkeypatch, capsys):
    sessions = []
    response = SimpleNamespace(
        text="回答",
        usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=7),
    )

    class Conversation:
        def send_message(self, message):
            self.messages.append(message)
            return response

        def __init__(self):
            self.messages = []
            sessions.append(self)

    class Chats:
        def create(self, **kwargs):
            assert kwargs == {
                "model": "test-model",
                "config": genai_types.GenerateContentConfig(temperature=0.2),
            }
            return Conversation()

    class Client:
        chats = Chats()

    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.genai, "Client", lambda api_key: Client())

    manager = ask_module.SessionManager(model="test-model", temperature=0.2)
    session_id = manager.start_session()
    assert manager.session_id == session_id
    assert manager.send_message("最初の質問") == "回答"
    assert manager.send_message("続きの質問") == "回答"
    with pytest.raises(RuntimeError, match="すでに開始"):
        manager.start_session()
    manager.end_session()

    assert sessions[0].messages == ["最初の質問", "続きの質問"]
    assert manager.session_id is None
    with pytest.raises(RuntimeError, match="セッションを開始"):
        manager.send_message("終了後の質問")
    assert capsys.readouterr().out.count("[応答時間]") == 2
