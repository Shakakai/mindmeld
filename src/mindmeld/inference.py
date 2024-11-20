import uuid
from pathlib import Path
from typing import List, Type, Optional, Tuple, Callable, Union, Dict, Any
from pydantic import BaseModel, Field, PrivateAttr
import openai
import instructor
import litellm
from mindmeld.pydantic_utils import pydantic_to_md
import json


class AIProvider(BaseModel):
    name: str
    api_key: Optional[str] = None
    api_key_env_var: Optional[str] = None
    api_base: Optional[str] = None


class AIModel(BaseModel):
    provider: AIProvider
    name: str


class RuntimeConfig(BaseModel):
    models: List[AIModel]
    eval_model: str
    default_model: str
    root_module: str = Field(description="The name of the root python module for this project")
    project_root: str = Field(description="file path to the project root. This is set at runtime based on the location of the .mindmeld file")
    dataset_dir: str = Field(description="The directory containing the dataset files")
    inference_config_dir: str = Field(description="The directory containing the inference configuration files")
    python_sources: List[str] = Field(description="The list of python source directories to search for inference modules", default_factory=list)

    _project_root: Path = PrivateAttr(default=None)
    _dataset_dir: Path = PrivateAttr(default=None)
    _inference_config_dir: Path = PrivateAttr(default=None)
    _python_sources: List[Path] = PrivateAttr(default=None)

    def resolve_paths(self, config_path: Path):
        self._project_root = (config_path.parent / self.project_root).resolve()
        self._dataset_dir = (self._project_root / self.dataset_dir).resolve()
        self._inference_config_dir = (self._project_root / self.inference_config_dir).resolve()
        self._python_sources = [(self._project_root / path).resolve() for path in self.python_sources]

    def get_project_root(self):
        return self._project_root

    def get_dataset_dir(self):
        return self._dataset_dir

    def get_inference_config_dir(self):
        return self._inference_config_dir

    def get_python_sources(self):
        return self._python_sources


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
InferenceType = Union[BaseModel, List[BaseModel]]


class DataEntry(BaseModel):
    id: str = Field(description="Unique identifier for the data entry", default_factory=uuid.uuid4)
    input: BaseModel
    expected: Optional[BaseModel] = None
    metadata: Optional[Dict[str, Any]] = None

    def model_dump(self):
        return {
            "id": self.id,
            "input": self.input.model_dump(),
            "expected": self.expected.model_dump(),
            "metadata": self.metadata
        }


class Dataset(BaseModel):
    entries: List[DataEntry] = Field(default_factory=list)
    _index: int = PrivateAttr(default=0)

    @classmethod
    def load(cls, runtime_config: RuntimeConfig, inference: "Inference", count: Optional[int] = None, id_list: Optional[List[str]] = None):
        dataset_file = runtime_config.get_dataset_dir() / f"{inference.id}.json"
        if not Path.exists(dataset_file):
            return []
        raw = json.load(open(dataset_file, "r"))
        result = cls.load_list(inference, raw)
        if id_list is not None:
            result = result.select(id_list)
        if count is not None:
            result = result.cap(count)
        return result

    @classmethod
    def load_list(cls, inference: "Inference", list_data: List[Dict[str, Any]]):
        result = Dataset()
        for entry in list_data:
            result.entries.append(DataEntry(
                id=entry["id"],
                input=inference.input_type(**entry["input"]),
                expected=inference.output_type(**entry["expected"]),
                metadata=entry["metadata"]
            ))
        return result

    @classmethod
    def save(cls, runtime_config: RuntimeConfig, inference: "Inference", dataset: "Dataset"):
        dataset_file = runtime_config.get_dataset_dir() / f"{inference.id}.json"
        raw_list = [entry.model_dump() for entry in dataset]
        json.dump(raw_list, open(dataset_file, "w"))

    def select(self, id_list: List[str]):
        result = Dataset()
        for entry in self:
            if entry.id in id_list:
                result.entries.append(entry)
        return result

    def cap(self, count: int):
        result = Dataset(entries=self.entries[:count])
        return result

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self) -> DataEntry:
        if self._index < len(self.entries):
            result = self.entries[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index: int):
        return self.entries[index]

    def __setitem__(self, key, value):
        self.entries[key] = value


class InferenceConfig(BaseModel):
    example_ids: Optional[List[str]] = Field(description="The ids of the examples to use for the inference", default=None)
    temperature: float = Field(default=1.0, description="The temperature to use for the model")
    eval_runs: int = 1
    eval_threshold: float = 1.0  # This will cause the evaluation to fail unless a perfect score is achieved,
    # users can then lower it to a more reasonable value
    prompt: Optional[str] = None
    model: Optional[str] = None


    @classmethod
    def config_file(cls, runtime_config: RuntimeConfig, inference: "Inference"):
        return runtime_config.get_inference_config_dir() / f"{inference.id}.json"
    
    @classmethod
    def load(cls, runtime_config:RuntimeConfig, inference: "Inference"):
        cf = cls.config_file(runtime_config, inference)
        if not Path.exists(cf):
            return cls.default()
        return cls(**json.load(open(cf, "r")))

    @classmethod
    def save(cls, runtime_config: RuntimeConfig, inference: "Inference", config: "InferenceConfig"):
        config_file = runtime_config.get_inference_config_dir() / f"{inference.id}.json"
        raw = config.model_dump()
        json.dump(raw, open(config_file, "w"))
    
    @classmethod
    def default(cls):
        return cls()


class Inference(BaseModel):
    id: str = Field(description="Unique identifier for the inference")
    instructions: str = Field(description="Instructions for the inference")
    input_type: Type[BaseModel] = Field(description="The Pydantic model type for the input of the inference")
    output_type: Type[BaseModel] = Field(description="The Pydantic model type for the output of the inference")
    metrics: List[MetricType] = Field(description="Metrics to evaluate the inference results", default_factory=list)
    examples: Optional[Dataset] = None
    config: Optional[InferenceConfig] = None # This is typically loaded from a file

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

    def get_temperature(self, runtime_config: RuntimeConfig) -> float:
        return self.get_config(runtime_config).temperature

    def get_eval_runs(self, runtime_config: RuntimeConfig) -> int:
        return self.get_config(runtime_config).eval_runs

    def get_eval_threshold(self, runtime_config: RuntimeConfig) -> float:
        return self.get_config(runtime_config).eval_threshold

    def get_config(self, runtime_config:RuntimeConfig):
        if self.config is None:
            self.config = InferenceConfig.load(runtime_config, self)
        return self.config

    def get_examples(self, runtime_config:RuntimeConfig):
        config = self.get_config(runtime_config)
        if config.example_ids is not None:
            return Dataset.load(runtime_config, self, id_list=config.example_ids)
        if self.examples is not None:
            return self.examples
        return Dataset.load(runtime_config, self, count=3)


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


def create_system_prompt(instructions: str, examples: Dataset):
    system_prompt = f"# Instructions\n{instructions}"
    if len(examples) > 0:
        system_prompt += "\n\n# Examples\n"
        count = 0
        for example in examples:
            count += 1
            system_prompt += f"\n\n## Example {count}\n{pydantic_to_md(example.input, level=2, label='Input')}"
            system_prompt += f"\n{pydantic_to_md(example.expected, level=2, label='Output')}"
    return system_prompt


class InferenceResult(BaseModel):
    result: Optional[BaseModel] = None
    system_prompt: str
    success: bool = False
    exception: Optional[str] = None

    def model_dump(self):
        return {
            "result": self.result.model_dump() if self.result is not None else None,
            "system_prompt": self.system_prompt,
            "success": self.success,
            "exception": self.exception
        }


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

    inference_config = inference.get_config(runtime_config)

    # create a system prompt if not provided
    if system_prompt is None:
        system_prompt = create_system_prompt(inference.instructions, inference.get_examples(runtime_config))
    if model_name is None:
        if inference_config.model is not None:
            model_name = inference_config.model
        elif test:
            model_name = runtime_config.eval_model
        else:
            model_name = runtime_config.default_model
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
            temperature=inference.get_temperature(runtime_config),
        )
        return InferenceResult(
            result=result,
            system_prompt=system_prompt,
            success=True
        )
    except Exception as e:
        return InferenceResult(system_prompt=system_prompt, exception=str(e))
