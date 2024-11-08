import pytest

from mindmeld.inference import Inference, RuntimeConfig, AIProvider, AIModel, run_inference
from mindmeld.metrics.echo import echo
from pydantic import BaseModel
from .conftest import echo_inference, EchoInput, EchoOutput


def test_run_inference(ollama_runtime_config, ollama_model_name):
    test_text= "Hello, world!"

    # Create test input
    echo_input = EchoInput(text=test_text)

    # Run the inference
    ir = run_inference(
        inference=echo_inference,
        input_data=echo_input,
        runtime_config=ollama_runtime_config,
        model_name=ollama_model_name
    )

    # Assert the result
    assert isinstance(ir.result, EchoOutput)
    assert ir.result.output == test_text

