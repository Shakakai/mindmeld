import pytest
from pydantic import BaseModel
from typing import List

from mindmeld.eval import eval_inference
from mindmeld.inference import Inference, MetricResultType, Dataset, DataEntry
from mindmeld.optimize import optimize_inference
from mindmeld.metrics.echo import echo


class TestType(BaseModel):
    text: str


def uppercase_metric(runtime_config, inference, prompt, input_data, output_data):
    is_uppercase = output_data.text == input_data.text.upper()
    return MetricResultType(
        metric_name=uppercase_metric.__name__,
        success=True,
        score=1.0 if is_uppercase else 0.0
    )


# Create a simple test inference that we can optimize
test_inference = Inference(
    id="test-optimization",
    version=1,
    instructions="Make text bigger",  # A dumb instruction that should convert to a better prompt
    input_type=TestType,
    output_type=TestType,
    metrics=[uppercase_metric],
    examples=Dataset(entries=[
        DataEntry(
            input=TestType(text="hello"),
            expected=TestType(text="HELLO")
        ),
        DataEntry(
            input=TestType(text="world"),
            expected=TestType(text="WORLD")
        ),
    ]),
    eval_runs=2,
    threshold=0.7,
    temperature=0.7
)


dataset = Dataset(entries=[
    DataEntry(input=TestType(text="test")),
    DataEntry(input=TestType(text="optimize")),
    DataEntry(input=TestType(text="machine learning")),
    DataEntry(input=TestType(text="artificial intelligence")),
    DataEntry(input=TestType(text="natural language processing")),
    DataEntry(input=TestType(text="deep learning")),
    DataEntry(input=TestType(text="neural networks"))
])


def test_optimize_inference(runtime_config, model_name):
    # Run optimization
    optimize_result = optimize_inference(
        inference=test_inference,
        dataset=dataset,
        runtime_config=runtime_config,
        inference_model_name=model_name,
        max_iterations=2
    )

    assert optimize_result.success

    history = optimize_result.history

    # Verify we have results
    assert history is not None
    assert len(history.prompt_history) > 0

    # Get the best prompt
    best_result = history.get_best_prompt()
    assert best_result is not None
    assert best_result.instructions != test_inference.instructions  # Should have improved the prompt
    improvement = history.get_improvement()
    assert improvement > 0.0  # Should have improved the score


def test_optimize_inference_no_improvement_needed(runtime_config, model_name):
    # Create an inference that's already optimal
    perfect_inference = Inference(
        id="perfect-test",
        version=1,
        instructions="Return the input text exactly as provided, unchanged.",
        input_type=TestType,
        output_type=TestType,
        metrics=[echo()],
        examples=Dataset(entries=[
            DataEntry(
                input=TestType(text="hello"),
                expected=TestType(text="hello")
            ),
        ]),
        eval_runs=2,
        threshold=0.7,
        temperature=0.0
    )

    test_data = [
        TestType(text="test", expected="test"),
    ]

    optimize_result = optimize_inference(
        inference=perfect_inference,
        dataset=dataset,
        runtime_config=runtime_config,
        inference_model_name=model_name,
        max_iterations=1
    )

    assert optimize_result.success

    history = optimize_result.history

    # Should stop early since the prompt is already good
    assert len(history.prompt_history) < 3
    
    best_result = history.get_best_prompt()
    assert best_result.result.score > 0.9  # Should have very high score


def test_optimize_inference_empty_input(runtime_config, model_name):
    with pytest.raises(ValueError):
        optimize_inference(
            inference=test_inference,
            dataset=Dataset(),
            runtime_config=runtime_config,
            inference_model_name=model_name,
            max_iterations=1
        )
