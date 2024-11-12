from typing import List, Type, Optional, Tuple, Callable, Union
from pydantic import BaseModel, Field
import openai
import instructor
import litellm
from mindmeld.pydantic_utils import pydantic_to_md


class AIProvider(BaseModel):
    name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None


class AIModel(BaseModel):
    provider: AIProvider
    name: str


class RuntimeConfig(BaseModel):
    models: List[AIModel]
    eval_model: str
    default_model: str


class InferenceConfig(BaseModel):
    id: str
    version: int
    ai_model_name: str
    system_prompt: str


class InferenceConfigCollection(BaseModel):
    inference_configs: List[InferenceConfig]


providers = {}


def get_client(model: AIModel):
    client = providers[model.provider.name] if model.provider.name in providers else None
    if client is None:
        if model.provider.name == "ollama":
            client = instructor.from_openai(
                openai.OpenAI(
                    base_url=model.provider.api_base,
                    api_key="ollama",  # required, but unused
                ),
                mode=instructor.Mode.JSON,
            )
        elif model.provider.name == "openai":
            client = instructor.from_openai(
                openai.OpenAI(
                    api_key=model.provider.api_key,
                ),
                mode=instructor.Mode.TOOLS,
            )
        else:
            client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.JSON)
        providers[model.provider.name] = client
    return client


class Metric(BaseModel):
    func: "MetricCallableType"
    weight: float = 1.0  # Must be between 0 and 1
    threshold: float = 0.0  # Can be used to set the threshold for an individual metric

    @property
    def name(self):
        return self.func.__name__


MetricType = Union[Metric, "MetricCallableType"]


class Inference(BaseModel):
    id: str
    version: int = Field(default=1)
    instructions: str
    input_type: Type[BaseModel]
    output_type: Type[BaseModel]
    metrics: List[MetricType] = Field(default_factory=list)
    examples: List[Tuple[BaseModel, BaseModel]] = Field(default_factory=list)
    temperature: float = Field(default=1.0, description="The temperature to use for the model")
    eval_runs: int = 1
    eval_threshold: float = 1.0  # This will cause the evaluation to fail unless a perfect score is achieved,
    # users can then lower it to a more reasonable value

    @property
    def standardized_metrics(self):
        result = []
        for metric in self.metrics:
            # we allow both Metric objects and MetricCallableType functions
            # Unify them to Metric objects
            if not isinstance(metric, Metric):
                metric = Metric(
                    name=metric.__name__,
                    func=metric
                )
            result.append(metric)
        return result


class MetricResultType(BaseModel):
    metric_name: str
    success: bool = False
    score: float = 0.0


MetricCallableType = Callable[
    [
        RuntimeConfig,
        Inference,  # Calling inference
        str,  # System prompt
        BaseModel,  # Input data
        BaseModel  # Output data
    ],
    MetricResultType
]
InferenceType = BaseModel | List[BaseModel]


def create_system_prompt(instructions: str, examples: List[Tuple[BaseModel, BaseModel]]):
    system_prompt = f"#Instructions\n{instructions}"
    if len(examples) > 0:
        system_prompt += "\n\n#Examples\n"
        count = 0
        for example in examples:
            count += 1
            system_prompt += f"\n\n##Example {count}\n{pydantic_to_md(example[0], level=2, label='Input')}"
            system_prompt += f"\n{pydantic_to_md(example[1], level=2, label='Output')}"
    return system_prompt


class InferenceResult(BaseModel):
    result: Optional[BaseModel] = None
    system_prompt: str
    success: bool = False
    exception: Optional[str] = None


def run_inference(
    inference: Inference,
    input_data: InferenceType,
    runtime_config: RuntimeConfig,
    model_name: str = None,
    system_prompt: Optional[str] = None,
    test: bool = False
) -> InferenceResult:
    # validate input data
    if not isinstance(input_data, inference.input_type):
        raise ValueError(f"Invalid input type: {type(input_data)}")

    # create a system prompt if not provided
    if system_prompt is None:
        system_prompt = create_system_prompt(inference.instructions, inference.examples)
    if model_name is None:
        model_name = runtime_config.eval_model if test else runtime_config.default_model
    if model_name is None:
        raise ValueError("Model name is required or a default model must be set in the runtime config")

    ai_model = None
    for model in runtime_config.models:
        if model.name == model_name:
            ai_model = model
            break

    if ai_model is None:
        raise Exception(f"Invalid model name: {model_name}")

    client = get_client(ai_model)
    message = pydantic_to_md(input_data)
    try:
        result = client.chat.completions.create(
            model=ai_model.name,
            response_model=inference.output_type,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=inference.temperature
        )
        return InferenceResult(
            result=result,
            system_prompt=system_prompt,
            success=True
        )
    except Exception as e:
        return InferenceResult(system_prompt=system_prompt, exception=str(e))
