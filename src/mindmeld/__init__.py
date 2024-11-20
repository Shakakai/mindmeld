__version__ = "0.2.0"

from mindmeld.eval import EvalMetricResult, EvalResult, eval_inference
from mindmeld.inference import (
    AIProvider,
    AIModel,
    RuntimeConfig,
    InferenceConfig,
    #InferenceConfigCollection,
    Metric,
    MetricType,
    InferenceType,
    DataEntry,
    Dataset,
    Inference,
    MetricResultType,
    MetricCallableType,
    InferenceResult,
    run_inference
)
from mindmeld.optimize import (
    PromptGeneration,
    OptimizationResult,
    PromptGenerationResult,
    PromptGenerationHistory,
    optimize_inference
)
from mindmeld.synthetic import generate_synthetic_input, generate_synthetic_pairs

__all__ = [
    # eval.py
    'EvalMetricResult',
    'EvalResult',
    'eval_inference',
    
    # inference.py
    'AIProvider',
    'AIModel', 
    'RuntimeConfig',
    'InferenceConfig',
    'Metric',
    'MetricType',
    'InferenceType',
    'DataEntry',
    'Dataset',
    'Inference',
    'MetricResultType',
    'MetricCallableType',
    'InferenceResult',
    'run_inference',
    
    # optimize.py
    'PromptGeneration',
    'OptimizationResult',
    'PromptGenerationResult', 
    'PromptGenerationHistory',
    'optimize_inference',
    
    # synthetic.py
    'generate_synthetic_input',
    'generate_synthetic_pairs'
]
