import pytest
from mindmeld.metrics.faithfulness import faithfulness
from mindmeld.inference import Inference, BaseModel


class InputData(BaseModel):
    context: str


class OutputData(BaseModel):
    answer: str


@pytest.fixture
def inference():
    return Inference(
        id="test_inference",
        instructions="Generate an answer based on the given context",
        input_type=InputData,
        output_type=OutputData
    )


@pytest.mark.parametrize("context,answer,expected_range", [
    ("The capital of France is Paris. It is known for the Eiffel Tower.", 
     "Paris is the capital of France and it has the Eiffel Tower.", 
     (0.8, 1.0)),
    ("The capital of France is Paris. It is known for the Eiffel Tower.", 
     "Paris is the capital of France and it has a population of 2 million.", 
     (0.4, 0.7)),
    ("The capital of France is Paris. It is known for the Eiffel Tower.", 
     "London is the capital of England and it has Big Ben.", 
     (0.0, 0.2)),
])
def test_faithfulness_various_inputs(runtime_config, model_name, inference, context, answer, expected_range):
    input_data = InputData(context=context)
    output_data = OutputData(answer=answer)

    metric_func = faithfulness(runtime_config, model_name)
    result = metric_func(inference, "Generate a faithful answer based on the context", input_data, output_data)

    assert isinstance(result, float)
    assert expected_range[0] <= result <= expected_range[1], f"Expected range: {expected_range}, Got: {result}"


@pytest.mark.parametrize("input_text", [
    "The sky is blue.",
    "John is angry.",
    "Africa is large.",
    "Football is better with an audience.",
])
def test_faithfulness_empty_answer(runtime_config, model_name, inference, input_text):
    input_data = InputData(context=input_text)
    output_data = OutputData(answer="")

    metric_func = faithfulness(runtime_config, model_name)
    result = metric_func(inference, "Generate a faithful answer based on the context", input_data, output_data)

    assert result == 0.0, f"Expected 0.0 for empty answer, got {result}"


@pytest.mark.parametrize("context", [
    "The Earth orbits around the Sun.",
    "The sky is blue.",
    "John is angry.",
    "Africa is large.",
    "Football is better with an audience.",
])
def test_faithfulness_identical_context_and_answer(runtime_config, model_name, inference, context):
    input_data = InputData(context=context)
    output_data = OutputData(answer=context)

    metric_func = faithfulness(runtime_config, model_name)
    result = metric_func(inference, "Generate a faithful answer based on the context", input_data, output_data)

    assert result == 1.0, f"Expected 1.0 for identical context and answer, got {result}"
