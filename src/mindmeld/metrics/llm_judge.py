from mindmeld.inference import Inference, MetricCallableType, InferenceType, run_inference, RuntimeConfig
from pydantic import BaseModel, Field


class JudgeInput(BaseModel):
    question: str = Field(description="The question to answer about the inference data provided")
    original_system_prompt: str = Field(
        description="System prompt for the original inference that transformed the input to the output")
    input_data: InferenceType = Field(description="The input for the inference")
    output_data: InferenceType = Field(description="The output from the inference")


class JudgeOutput(BaseModel):
    answer: float = Field(description="An answer to the question on a scale from 0.0 to 1.0", ge=0.0, le=1.0)
    reasoning: str = Field(description="Provide in-depth reasoning for your answer")


llm_judge_inference = Inference(
    id="llm_judge",
    instructions="You must answer the question below using the tool provided. "
                 "If the question is yes/no, the answer should be a 1 or 0. "
                 "If the question is open-ended, provide an answer on a scale from 1 to 0.",
    input_type=JudgeInput,
    output_type=JudgeOutput
)


def llm_judge(
        instruction: str
) -> MetricCallableType:
    """
    A metric that uses an LLM to judge a true/false question about the result of an inference.
    Example: Does the output contain profanity?

    Args:
        model_name: (string): The name of the LLM model to use for judging.
        instruction (string): What is being judged from the provided data.

    Returns:
        Callable: A function that takes an Inference, system prompt, input data,
                  and output data, and returns a float score of 0 or 1.
    """

    def __impl__(
            runtime_config: RuntimeConfig,
            inference: Inference,
            system_prompt: str,
            input_data: BaseModel,
            output_data: BaseModel
    ) -> float:
        judge_input = JudgeInput(
            question=instruction,
            original_system_prompt=system_prompt,
            input_data=input_data,
            output_data=output_data
        )
        judge_result = run_inference(llm_judge_inference, judge_input, runtime_config, test=True)
        return judge_result.result.answer

    __impl__.__name__ = f"llm_judge:{instruction}"
    return __impl__