from mindmeld.inference import Inference, InferenceType, RuntimeConfig, run_inference, Metric

from pydantic import BaseModel, Field
from typing import Dict, Optional, List


class EvalMetricResult(BaseModel):
    name: str
    scores: List[float] = Field(default_factory=list)
    mean_score: float = 0.0
    max_score: Optional[float] = None
    min_score: Optional[float] = None
    weighted_score: float = 0.0
    weight: float = 1.0
    threshold: float = 0.0

    def update(self):
        self.min_score = min(self.scores)
        self.max_score = max(self.scores)
        self.mean_score = sum(self.scores) / len(self.scores)
        self.weighted_score = self.mean_score * self.weight


class EvalResult(BaseModel):
    score: float = 0.0
    metrics: List[EvalMetricResult] = Field(default_factory=list)
    success: bool = False
    threshold: float = 1.0

    def update(self):
        self.success = True
        for metric in self.metrics:
            metric.update()
            if metric.mean_score < metric.threshold:
                self.success = False

        total_score = sum([m.weighted_score for m in self.metrics])
        total_weight = sum([m.weight for m in self.metrics])
        self.score = total_score / total_weight

        if self.score < self.threshold:
            self.success = False


def eval_inference(
    inference: Inference, 
    input_data: InferenceType, 
    runtime_config: RuntimeConfig, 
    model_name: str = None,
    system_prompt: Optional[str] = None
) -> EvalResult:
    runs_left = inference.eval_runs
    eval_metrics = inference.standardized_metrics
    eval_result = EvalResult(threshold=inference.eval_threshold)
    eval_metric_results = {}
    while runs_left > 0:
        runs_left -= 1
        inference_result = run_inference(inference, input_data, runtime_config, model_name, system_prompt)
        for metric in eval_metrics:
            if metric.name not in eval_metric_results:
                eval_metric_results[metric.name] = EvalMetricResult(
                    name=metric.name,
                    weight=metric.weight,
                    threshold=metric.threshold
                )
                eval_result.metrics.append(eval_metric_results[metric.name])
            eval_metric_result = eval_metric_results[metric.name]
            metric_result = metric.func(
                runtime_config,
                inference,
                inference_result.system_prompt,
                input_data,
                inference_result.result
            )
            eval_metric_result.scores.append(metric_result)
    eval_result.update()
    return eval_result
