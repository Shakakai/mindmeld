import pytest
from mindmeld.metrics.answer_similarity import answer_similarity
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
    ("What is the capital of France?", "Paris is the capital of France.", (0.6, 1.0)),
    ("What is the capital of France?", "The capital of France is Paris.", (0.6, 1.0)),
    ("Who invented the telephone?", "Alexander Graham Bell invented the telephone.", (0.6, 1.0)),
    ("What is the capital of France?", "London is seat of power for England.", (0.0, 0.6)),
    ("Who invented the telephone?", "Thomas Edison created the light bulb.", (0.0, 0.6)),
])
def test_answer_similarity_various_inputs(runtime_config, model_name, inference, question, answer, expected_range):
    input_data = InputData(question=question)
    output_data = OutputData(answer=answer)

    metric_func = answer_similarity(runtime_config, model_name)
    result = metric_func(inference, "Answer the question accurately", input_data, output_data)

    assert isinstance(result, float)
    assert expected_range[0] <= result, f"Expected value above {expected_range[0]}, Got: {result}"
    assert result <= expected_range[1], f"Expected value below {expected_range[1]}, Got: {result}"


@pytest.mark.parametrize("question", [
    "What is the capital of France?",
    "Who invented the telephone?",
    "What is the largest planet in our solar system?",
])
def test_answer_similarity_empty_answer(runtime_config, model_name, inference, question):
    input_data = InputData(question=question)
    output_data = OutputData(answer="")

    metric_func = answer_similarity(runtime_config, model_name)
    result = metric_func(inference, "Answer the question accurately", input_data, output_data)

    assert result <= 0.2, f"Expected 0.2 of less for empty answer, got {result}"

