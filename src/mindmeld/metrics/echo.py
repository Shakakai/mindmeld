from mindmeld.inference import Inference, MetricCallableType, RuntimeConfig, MetricResultType
from pydantic import BaseModel

from mindmeld.pydantic_utils import pydantic_to_md, pydantic_to_vs


def echo() -> MetricCallableType:
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
        input_md = pydantic_to_vs(input_data)
        output_md = pydantic_to_vs(output_data)
        result = 1.0 if input_md == output_md else 0.0
        return MetricResultType(metric_name=__impl__.__name__, success=True, score=result)

    __impl__.__name__ = "echo"
    return __impl__
