from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from mindmeld.eval import EvalResult
from mindmeld.inference import Inference, InferenceType, RuntimeConfig, run_inference, Dataset
from mindmeld.eval import eval_inference


class PromptGeneration(BaseModel):
    instructions: str = Field(..., description="The instructions for the prompt")
    reasoning: str = Field(..., description="The reasoning for the prompt generation")


class OptimizationResult(BaseModel):
    eval_results: List[EvalResult] = Field(default_factory=list)
    score: float = 0.0
    metric_scores: Dict[str, List[float]] = Field(default_factory=dict)
    metric_weights: Dict[str, float] = Field(default_factory=dict)

    def update(self):
        self.metric_scores = {}
        self.metric_weights = {}
        total_weight = 0.0
        self.score = 0.0
        for eval_result in self.eval_results:
            for metric in eval_result.metrics:
                if metric.name not in self.metric_scores:
                    self.metric_scores[metric.name] = []
                    self.metric_weights[metric.name] = metric.weight
                    total_weight += metric.weight
                self.metric_scores[metric.name].append(metric.weighted_score)
        for metric_name, metric_values in self.metric_scores.items():
            self.metric_scores[metric_name] = sum(metric_values) / len(metric_values)
            self.score += self.metric_scores[metric_name] * self.metric_weights[metric_name]
        self.score /= total_weight


class PromptGenerationResult(BaseModel):
    instructions: str = Field(..., description="The instructions for the prompt")
    result: OptimizationResult = Field(..., description="The scores from testing this prompt")


class PromptGenerationHistory(BaseModel):
    iteration_index: int = Field(..., description="The zero-based index of the current iteration")
    max_iterations: int = Field(..., description="The maximum number of iterations to run")
    prompt_history: List[PromptGenerationResult] = Field(..., description="The history of prompts generated and tested")

    def get_best_prompt(self) -> PromptGenerationResult:
        """
        Returns the PromptGenerationResult with the highest evaluation score.
        
        Returns:
            PromptGenerationResult: The result containing the best performing prompt and its test results
        """
        if not self.prompt_history:
            raise ValueError("No prompts in history")
        
        return max(self.prompt_history, key=lambda x: x.result.score)

    def get_improvement(self) -> float:
        """
        Returns the improvement in score from the first prompt to the best prompt.

        Returns:
            float: The improvement in score
        """
        if not self.prompt_history:
            raise ValueError("No prompts in history")

        best_prompt = self.get_best_prompt()
        first_prompt = self.prompt_history[0]
        return best_prompt.result.score - first_prompt.result.score


prompt_optimization_inference = Inference(
    id="prompt-optimization",
    version=1,
    instructions="Optimize the prompt for the given inference. Do not repeat a prompt that was previously tested.",
    input_type=PromptGenerationHistory,
    output_type=PromptGeneration
)


class OptimizeResult(BaseModel):
    success: bool = False
    best_prompt: Optional[str] = None
    improvement: Optional[float] = None
    history: Optional[PromptGenerationHistory] = None


def optimize_inference(
    inference: Inference,
    dataset: Dataset,
    runtime_config: RuntimeConfig,
    inference_model_name: Optional[str] = None,
    optimize_model_name: Optional[str] = None,
    max_iterations: int = 10
) -> OptimizeResult:
    if dataset is None or len(dataset) == 0:
        raise ValueError("No input data provided")

    if inference_model_name is None:
        inference_model_name = runtime_config.default_model
    
    if optimize_model_name is None:
        optimize_model_name = runtime_config.eval_model or runtime_config.default_model
    
    result = eval_inference(
        inference,
        dataset[0].input,
        runtime_config,
        inference_model_name
    )
    first_optimization_result = OptimizationResult()
    first_optimization_result.eval_results.append(result)
    first_optimization_result.update()
    initial_prompt_generation_result = PromptGenerationResult(
        instructions=inference.instructions,
        result=first_optimization_result
    )
    iteration_index = 0
    history = PromptGenerationHistory(
        iteration_index=iteration_index,
        max_iterations=max_iterations,
        prompt_history=[initial_prompt_generation_result]
    )
    while iteration_index < max_iterations:
        optimization_result = OptimizationResult()
        prompt_gen = run_inference(
            prompt_optimization_inference,
            history,
            runtime_config,
            optimize_model_name
        )
        for item in dataset:
            test_result = eval_inference(
                inference,
                item.input,
                runtime_config,
                inference_model_name,
                system_prompt=prompt_gen.result.instructions
            )
            optimization_result.eval_results.append(test_result)
        optimization_result.update()
        history.prompt_history.append(PromptGenerationResult(
            instructions=prompt_gen.result.instructions,
            result=optimization_result
        ))
        if optimization_result.score == 1.0:
            break
        iteration_index += 1

    return OptimizeResult(
        success=True,
        best_prompt=history.get_best_prompt().instructions,
        improvement=history.get_improvement(),
        history=history
    )
