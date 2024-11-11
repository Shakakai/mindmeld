import pytest
from pydantic import BaseModel
from mindmeld.inference import AIProvider, AIModel, RuntimeConfig, Inference, run_inference
from mindmeld.metrics.echo import echo


def test_ollama_runtime_config(ollama_provider, ollama_runtime_config):
    # Test provider configuration
    assert ollama_provider.name == "ollama"
    assert ollama_provider.api_base == "http://localhost:11434/v1"
    assert ollama_provider.api_key is None

    # Test runtime config
    assert len(ollama_runtime_config.models) == 2
    
    model = ollama_runtime_config.models[0]
    assert isinstance(model, AIModel)
    assert model.provider == ollama_provider
    assert model.name == "llama3.2:1b"


