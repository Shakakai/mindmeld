from mindmeld.inference import (
    Inference, MetricCallableType,
    run_inference, RuntimeConfig,
    MetricResultType, DataEntry, Dataset
)
from pydantic import BaseModel, Field


class AnswerRelevanceInput(BaseModel):
    system_prompt: str = Field(description="The original system prompt that generated the output_data")
    input_data: BaseModel = Field(description="The original question asked")
    output_data: BaseModel = Field(description="The answer generated based on the question")


class AnswerRelevanceOutput(BaseModel):
    relevance_score: float = Field(description="A score between 0 and 1 indicating the relevance of the answer")
    reasoning: str = Field(description="Explanation for the given score")


answer_relevance_inference = Inference(
    id="answer_relevance",
    version=1,
    instructions="""
    Evaluate the relevance of the given output for the system prompt and input data.
    Provide a relevance score between 0 and 1, where 0 indicates no relevance and 1 indicates high relevance.
    There may be context missing from the system prompt that is necessary to evaluate the relevance.
    Make reasonable assumptions about any context that is missing.
    """,
    input_type=AnswerRelevanceInput,
    output_type=AnswerRelevanceOutput,
    temperature=0.0,
    examples=Dataset(entries=[
        DataEntry(
            input=AnswerRelevanceInput(
                system_prompt="Answer the question based on the data provided in the input",
                input_data={"text": "# Context: The capital of France is Paris. "
                                    "\n\n# Question: What is the capital of France?"},
                output_data={"answer": "The capital of France is Paris."}
            ),
            expected=AnswerRelevanceOutput(
                relevance_score=1.0,
                reasoning="The output is a direct answer to the question based on the context provided in the input."
            )
        ),
        DataEntry(
            input=AnswerRelevanceInput(
                system_prompt="Answer the question based on the data provided in the input",
                input_data={ "text": "# Context: The capital of France is Paris. "
                                     "\n\n# Question: What is the capital of France?"},
                output_data={"answer": "The capital of France is London."}
            ),
            expected=AnswerRelevanceOutput(
                relevance_score=0.0,
                reasoning="The output is a direct answer to the question, "
                          "but it is incorrect based on the context provided."
            )
        )
    ])
)


def answer_relevance() -> MetricCallableType:
    """
    A metric that evaluates the relevance of an answer to a given question using question generation and similarity comparison.

    Args:
        runtime_config (RuntimeConfig): Configuration for the runtime environment.
        model_name (str): Name of the model to use for evaluation.

    Returns:
        Callable: A function that takes an Inference, system prompt, input data,
                  and output data, and returns a float score between 0 and 1.
    """

    def __impl__(
        runtime_config: RuntimeConfig,
        inference: Inference,
        system_prompt: str,
        input_data: BaseModel,
        output_data: BaseModel
    ) -> MetricResultType:
        # Run the answer relevance inference
        relevance_input = AnswerRelevanceInput(
            system_prompt=system_prompt,
            input_data=input_data,
            output_data=output_data
        )
        relevance_output = run_inference(answer_relevance_inference, relevance_input, runtime_config, test=True)
        if not relevance_output.success:
            return MetricResultType(
                metric_name=__impl__.__name__,
                success=False,
            )
        return MetricResultType(
            metric_name=__impl__.__name__,
            success=True,
            score=relevance_output.result.relevance_score
        )

    __impl__.__name__ = "answer_relevance"
    return __impl__
