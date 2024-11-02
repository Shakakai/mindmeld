import pytest
from mindmeld.metrics.answer_relevance import answer_relevance
from mindmeld.inference import Inference, BaseModel


class InputData(BaseModel):
    question: str


class OutputData(BaseModel):
    answer: str


@pytest.fixture
def inference():
    return Inference(
        id="test_inference",
        instructions="Answer the question",
        input_type=InputData,
        output_type=OutputData
    )


@pytest.mark.parametrize("question,answer,expected_range", [
    ("What is the capital of France?", "Paris is the capital of France.", (0.7, 1.0)),
    ("What is the capital of France?", "The capital of France is Paris.", (0.7, 1.0)),
    ("Who's free tonight?", "I am.", (0.3, 1.0)),
    ("Who's free tonight?", "I am not in jail.", (0.0, 0.7)),
    ("What is the capital of France?", "The capital of France is Berlin.", (0.0, 0.3)),
    ("What is the capital of France?", "The weather in Paris is nice.", (0.0, 0.3)),
    ("What is the capital of France?", "Elephants are large mammals.", (0.0, 0.3)),
])
def test_answer_relevance_various_inputs(runtime_config, model_name, inference, question, answer, expected_range):
    input_data = InputData(question=question)
    output_data = OutputData(answer=answer)

    metric_func = answer_relevance(runtime_config, model_name)
    result = metric_func(inference, "Answer the question accurately", input_data, output_data)

    assert isinstance(result, float)
    assert expected_range[0] <= result, f"Result should be larger than {expected_range[0]}"
    assert result <= expected_range[1], f"Result should be smaller then {expected_range[1]}"

