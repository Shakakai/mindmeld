from mindmeld.inference import Inference, run_inference, RuntimeConfig
from typing import Type, Optional, List
from pydantic import BaseModel, Field


def generate_synthetic_input(
    runtime_config: RuntimeConfig,
    inference: Inference,
    count: int = 10
):
    class SyntheticData(BaseModel):
        data: List[inference.input_type] = Field(default_factory=list)

    synthetic_data_inference = Inference(
        id="synthetic_data_generation",
        instructions=f"""
    You are a test data generator. Generate test data using the tool provided.
    Be sure not to duplicate any of the data already provided, shown below.
        """,
        input_type=SyntheticData,
        output_type=inference.input_type
    )

    result = SyntheticData()
    while count > 0:
        inference_result = run_inference(synthetic_data_inference, result, runtime_config)
        if not inference_result.success:
            raise Exception(f"Failed to generate synthetic data: {inference_result}")
        result.data.append(inference_result.result)
        count -= 1
    return result.data


def generate_synthetic_pairs(
    runtime_config: RuntimeConfig,
    inference: Inference,
    count: int = 10
):
    syn_inputs = generate_synthetic_input(runtime_config, inference, count)
    results = []
    for syn_input in syn_inputs:
        inference_result = run_inference(inference, syn_input, runtime_config)
        if not inference_result.success:
            raise Exception(f"Failed to generate synthetic data: {inference_result}")
        results.append((syn_input, inference_result.result))
    return results
