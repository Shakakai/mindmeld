import pytest

from mindmeld.inference import Inference, RuntimeConfig, AIProvider, AIModel, run_inference
from pydantic import BaseModel
from tests.conftest import EchoInput, EchoOutput


def test_run_inference(runtime_config, model_name, echo_inference):
    test_text= "Hello, world!"

    # Create test input
    echo_input = EchoInput(text=test_text)

    # Run the inference
    result = run_inference(
        inference=echo_inference,
        input_data=echo_input,
        runtime_config=runtime_config,
        model_name=model_name
    )

    # Assert the result
    assert isinstance(result, EchoOutput)
    assert result.result == test_text

