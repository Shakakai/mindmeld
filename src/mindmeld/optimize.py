
class PromptGeneration(BaseModel):
    instructions: str = Field(..., description="The instructions for the prompt")
    reasoning: str = Field(..., description="The reasoning for the prompt generation")


class PromptGenerationResult(BaseModel):
    instructions: str = Field(..., description="The instructions for the prompt")
    test_result: TestResult = Field(..., description="The scores from testing this prompt")


class PromptGenerationHistory(BaseModel):
    iteration_index: int = Field(..., description="The zero-based index of the current iteration")
    max_iterations: int = Field(..., description="The maximum number of iterations to run")
    prompt_history: List[PromptGenerationResult] = Field(..., description="The history of prompts generated and tested")


prompt_optimization_inference = Inference(
    id="prompt-optimization",
    version=1,
    instructions="Optimize the prompt for the given inference. Do not repeat a prompt that was previously tested.",
    input_type=PromptGenerationHistory,
    output_type=PromptGeneration
)


def optimize_inference(
    inference: Inference, 
    input_data: List[InferenceType], 
    runtime_config: RuntimeConfig, 
    test_model_name: str,
    inference_model_name: str,
    max_iterations: int = 10
) -> PromptGenerationHistory:
    result = test_inference(
        inference, 
        input_data, 
        runtime_config,
        inference_model_name
    )
    initial_prompt_generation_result = PromptGenerationResult(
        instructions=inference.instructions,
        test_result=result
    )
    
    iteration_index = 0
    history = PromptGenerationHistory(
        iteration_index=iteration_index,
        max_iterations=max_iterations,
        prompt_history=[initial_prompt_generation_result]
    )
    while iteration_index < max_iterations:
        test_results = []
        prompt_gen = run_inference(
            prompt_optimization_inference, 
            history, 
            runtime_config, 
            test_model_name
        )
        for i in input_data:
            test_result = test_inference(
                inference, 
                i, 
                runtime_config, 
                inference_model_name, 
                prompt_gen.instructions
            )
            test_results.append(test_result)
        
        avg_test_results = {}
        for test_result in test_results:
            for metric_name, metric_value in test_result.metrics.items():
                if metric_name not in avg_test_results:
                    avg_test_results[metric_name] = []
                avg_test_results[metric_name].append(metric_value)
        
        for metric_name, metric_values in avg_test_results.items():
            avg_test_results[metric_name] = sum(metric_values) / len(metric_values)

        score = 0.0
        max_score = 0.0
        for metric_name, metric_values in avg_test_results.items():
            score += metric_values * inference.metrics[metric_name].weight
            max_score += inference.metrics[metric_name].weight
        
        score /= max_score

        history.prompt_history.append(PromptGenerationResult(
            instructions=prompt_gen.instructions,
            test_result=TestResult(
                score=score,
                metrics=avg_test_results
            )
        ))
        iteration_index += 1
    
    return history
