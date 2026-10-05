import pytest
import os
from idrift.llm_client import LLMClient, ChatResult

def test_llm_client_missing_key(monkeypatch):
    monkeypatch.delenv("TEST_KEY", raising=False)
    client = LLMClient(model="m", api_key_env="TEST_KEY")
    with pytest.raises(RuntimeError, match="missing env var TEST_KEY"):
        client._get_client()

def test_llm_client_success(mocker, monkeypatch):
    monkeypatch.setenv("DUMMY_KEY", "secret")
    
    mock_openai = mocker.patch("idrift.llm_client.OpenAI")
    mock_client = mock_openai.return_value
    
    mock_choice = mocker.Mock()
    mock_choice.message.content = "hello"
    mock_choice.finish_reason = "stop"
    
    mock_usage = mocker.Mock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    
    mock_resp = mocker.Mock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = mock_usage
    
    mock_client.chat.completions.create.return_value = mock_resp
    
    client = LLMClient(model="dummy-model", api_key_env="DUMMY_KEY")
    res = client.chat([{"role": "user", "content": "hi"}], temperature=0.0)
    
    assert res.text == "hello"
    assert res.usage_in == 10
    assert res.usage_out == 5
    assert res.model == "dummy-model"
    
    mock_client.chat.completions.create.assert_called_once_with(
        model="dummy-model",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=512,
    )
    
    # Check key passed
    mock_openai.assert_called_once_with(api_key="secret", timeout=60.0)
