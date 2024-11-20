import pytest
from typing_extensions import runtime

from mindmeld.metrics.llm_judge import llm_judge
from mindmeld.inference import Inference, BaseModel, MetricResultType


class InputData(BaseModel):
    text: str


class OutputData(BaseModel):
    result: str


@pytest.fixture
def inference():
    return Inference(
        id="test_inference",
        instructions="Process the input text",
        input_type=InputData,
        output_type=OutputData
    )


@pytest.mark.parametrize("input_text,output_text,instruction,lower_bound,upper_bound", [
    ("Hello world", "HELLO WORLD", "Is the output text in all uppercase?", 0.7, 1.0),
    ("Hello world", "hello world", "Is the output text in all uppercase?", 0.0, 0.3),
    ("The quick brown fox", "The quick brown fox jumps over the lazy dog", "Does the output contain more words than the input?", 0.7, 1.0),
    ("The quick brown fox", "The quick fox", "Does the output contain more words than the input?", 0.0, 0.3),
])
def test_llm_judge_various_inputs(
        runtime_config,
        model_name,
        inference,
        input_text,
        output_text,
        instruction,
        lower_bound,
        upper_bound
):
    input_data = InputData(text=input_text)
    output_data = OutputData(result=output_text)

    metric_func = llm_judge(instruction)
    result = metric_func(
        runtime_config,
        inference,
        "Process the input text as instructed",
        input_data,
        output_data
    )

    assert isinstance(result, MetricResultType)
    assert lower_bound <= result.score <= upper_bound, f"Expected between {lower_bound} and {upper_bound}, but got {result.score}"


def test_llm_judge_empty_output(runtime_config, model_name, inference):
    input_data = InputData(text="Hello world")
    output_data = OutputData(result="")
    instruction = "Is the output empty?"

    metric_func = llm_judge(instruction)
    result = metric_func(runtime_config, inference, "Process the input text", input_data, output_data)

    assert isinstance(result, MetricResultType)
    assert 0.7 <= result.score <= 1.0, f"Expected 1.0 for empty output, got {result}"


def test_llm_judge_identical_input_output(runtime_config, model_name, inference):
    text = "The sky is blue."
    input_data = InputData(text=text)
    output_data = OutputData(result=text)
    instruction = "Is the output identical to the input?"

    metric_func = llm_judge(instruction)
    result = metric_func(runtime_config, inference, "Process the input text", input_data, output_data)

    assert isinstance(result, MetricResultType)
    assert 0.7 <= result.score <= 1.0, f"Expected 1.0 for identical input and output, got {result}"


def test_llm_judge_complex_instruction(runtime_config, model_name, inference):
    input_data = InputData(text="The capital of France is Paris.")
    output_data = OutputData(result="Paris is the capital city of France, known for the Eiffel Tower.")
    instruction = "Does the output contain information not present in the input while still being relevant to the input?"

    metric_func = llm_judge(instruction)
    result = metric_func(runtime_config, inference, "Expand on the input text with relevant information", input_data, output_data)

    assert isinstance(result, MetricResultType)
    assert 0.7 <= result.score <= 1.0, f"Expected 1.0 for output with additional relevant information, got {result}"
