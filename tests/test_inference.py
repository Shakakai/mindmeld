import pytest

from mindmeld.inference import run_inference
from .conftest import echo_inference, EchoType


def test_run_inference(runtime_config, model_name):
    test_text= "Hello, world!"

    # Create test input
    echo_input = EchoType(text=test_text)

    # Run the inference
    ir = run_inference(
        inference=echo_inference,
        input_data=echo_input,
        runtime_config=runtime_config,
        model_name=model_name
    )

    # Assert the result
    assert isinstance(ir.result, EchoType)
    assert ir.result.text == test_text

