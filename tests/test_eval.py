from mindmeld.eval import eval_inference
from .conftest import echo_inference, EchoType


def test_echo_eval(runtime_config):
    test_text = "Hello, world!"

    # Create test input
    echo_input = EchoType(text=test_text)

    # Run the inference
    eval_result = eval_inference(
        inference=echo_inference,
        input_data=echo_input,
        runtime_config=runtime_config
    )

    # Assert the result
    assert eval_result.success
    assert eval_result.score == 1.0
