from mindmeld.inference import Inference, MetricCallableType, RuntimeConfig, MetricResultType
from pydantic import BaseModel
import math


def token_efficiency_score(token_count, midpoint, steepness=0.01):
    # Use a sigmoid function to calculate the score
    return 1 / (1 + math.exp(steepness * (token_count - midpoint)))


def shorten_prompt_length(max_length=500) -> MetricCallableType:
    """
    A metric function that evaluates the length of the system prompt.

    This function returns a closure that calculates a score based on how well
    the system prompt length stays within a specified maximum length. The score
    is normalized between 0 and 1, where:
    - 1.0 indicates the prompt is empty (optimal brevity)
    - 0.0 indicates the prompt exceeds the maximum length
    - Values between 0 and 1 represent how close the prompt is to the max length

    Args:
        max_length (int): The maximum desired length for the system prompt.
                          Defaults to 1000 characters.

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
        prompt_length = len(system_prompt)
        score = token_efficiency_score(prompt_length, midpoint=max_length)
        return MetricResultType(metric_name=__impl__.__name__, success=True, score=score)

    __impl__.__name__ = "shorten_prompt_length"
    return __impl__
