import os

import pytest

import litellm


@pytest.fixture
def service_key():
    key = os.environ.get("AICORE_SERVICE_KEY")
    if not key:
        pytest.skip("AICORE_SERVICE_KEY not set")
    return key


def test_sap_live_basic_completion(service_key):
    response = litellm.completion(
        model="sap/gpt-4o",
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        api_key=service_key,
        max_tokens=10,
    )
    assert response.choices[0].message.content is not None
    assert response.usage.total_tokens > 0


def test_sap_live_reasoning_effort_o_model(service_key):
    response = litellm.completion(
        model="sap/openai--o4-mini",
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        api_key=service_key,
        max_completion_tokens=100,
        reasoning_effort="low",
    )
    assert response.choices[0].message.content is not None
    assert response.usage.total_tokens > 0
