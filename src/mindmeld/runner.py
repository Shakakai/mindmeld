from typing import List, Type, Optional, Tuple
from pydantic import BaseModel
import instructor
from openai import OpenAI
from .pydantic_utils import pydantic_to_md


class AIProvider(BaseModel):
    name: str
    # Eventually we'll add this: api_key: str


class AIModel(BaseModel):
    provider: AIProvider
    name: str


class RuntimeConfig(BaseModel):
    models: List[AIModel]


class InferenceConfig(BaseModel):
    id: str
    version: int
    model_name: str
    system_prompt: str


class InferenceConfigCollection(BaseModel):
    inference_configs: List[InferenceConfig]


class Metric:

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        if weight <= 0 or weight > 1:
            raise ValueError("Weight must be between 0 and 1")
        self.weight = weight
        self.result = None
        self.error = None

    def evaluate(
        self,
        inference: "Inference",
        input: BaseModel,
        output: BaseModel
    ):
        pass


providers = {}


def get_client(model: AIModel):
    client = providers[model.provider.name] if model.provider.name in providers else None
    if client is None:
        if model.provider.name == "openai":
            client = instructor.patch(OpenAI())
        else:
            raise ValueError(f"Invalid AI provider: {model.provider.name}")
        providers[model.provider.name] = client
    return client


class Inference(BaseModel):
    id: str
    version: int
    instructions: str
    input_type: Type[BaseModel]
    output_type: Type[BaseModel]
    metrics: List[Metric]
    examples: List[Tuple[BaseModel, BaseModel]]


InferenceType = BaseModel | List[BaseModel]


def run_inference(
    inference: Inference,
    input_data: InferenceType,
    runtime_config: RuntimeConfig,
    model_name: str = None,
    system_prompt: Optional[str] = None
):
    if system_prompt is None:
        system_prompt = f"#Instructions\n{inference.instructions}"
        if len(inference.examples) > 0:
            system_prompt += "\n\n#Examples\n"
            for example in inference.examples:
                system_prompt += f"\n\n##Example\n ###Input\n{example[0]}\n ###Output\n{example[1]}"
    if not isinstance(input_data, inference.input_type):
        raise ValueError(f"Invalid input type: {type(input)}")
    ai_model = None
    for model in runtime_config.models:
        if model.name == model_name:
            ai_model = model
            break
    if ai_model is None:
        raise Exception(f"Invalid model name: {model_name}")
    client = get_client(ai_model)
    message = pydantic_to_md(input_data)
    return client.chat.completions.create(
        model=ai_model.name,
        response_model=inference.output_type,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
    )


""" Need to finish this.
def main(
    runtime_config: RuntimeConfig,
    module_or_inference_reference: str,
    mode: Literal["test", "optimize", "serve"],
    inference_config: Optional[InferenceConfig] = None
):
    import_path = module_or_inference_reference.split(".")
    py_module_path = ".".join(import_path[:-1])
    py_module = importlib.import_module(py_module_path)
    module_or_inference = getattr(py_module, import_path[-1])
    inference_list = []
    if isinstance(module_or_inference, Inference):
        inference_list.append(module_or_inference)
    elif isinstance(module_or_inference, MMModule):
        inference_list += module_or_inference.get_inference_list()
    else:
        raise ValueError(f"Invalid module or inference reference: {module_or_inference}")
    for inference in inference_list:
        inference_function(model)
"""
