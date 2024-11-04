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
    assert len(ollama_runtime_config.models) == 1
    
    model = ollama_runtime_config.models[0]
    assert isinstance(model, AIModel)
    assert model.provider == ollama_provider
    assert model.name == "llama3.2:1b"


def test_ollama_inference(ollama_runtime_config):
    # Define simple input/output models
    class Input(BaseModel):
        text: str

    class Output(BaseModel):
        response: str

    # Create inference config
    inference = Inference(
        id="test_ollama",
        instructions="Always respond with 'Hello World'. Do not respond with anything else.",
        input_type=Input,
        output_type=Output,
        metrics=[
            echo(),
        ],
        examples=[
            (Input(text="Hi"), Output(response="Hello World")),
            (Input(text="How are you?"), Output(response="Hello World")),
        ]
    )

    test_input = Input(text="Hi")
    response = run_inference(
        inference=inference,
        input_data=test_input,
        runtime_config=ollama_runtime_config,
        model_name="llama3.2:1b"
    )

    assert isinstance(response.result, Output)
    assert isinstance(response.result.response, str)
    assert len(response.result.response) > 0
    assert isinstance(response.system_prompt, str)

