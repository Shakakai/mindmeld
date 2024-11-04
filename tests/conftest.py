import pytest
from pydantic import BaseModel

from mindmeld.inference import RuntimeConfig, AIModel, AIProvider, Inference
from mindmeld.metrics.echo import echo
#import logging


#logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def model_name():
    return "gpt-4o"


@pytest.fixture
def runtime_config(model_name):
    return RuntimeConfig(
        models=[
            AIModel(
                provider=AIProvider(name="openai"),
                name=model_name
            )
        ],
        eval_model=model_name,
        default_model=model_name
    )

@pytest.fixture
def ollama_model_name():
    return "llama3.2:1b"


@pytest.fixture
def ollama_provider():
    return AIProvider(
        name="ollama", 
        api_base="http://localhost:11434/v1"
    )


@pytest.fixture
def ollama_runtime_config(ollama_provider, ollama_model_name, model_name):
    return RuntimeConfig(
        models=[
            AIModel(
                provider=ollama_provider, 
                name=ollama_model_name
            ),
            AIModel(
                provider=AIProvider(name="openai"),
                name=model_name
            )
        ],
        eval_model=model_name,
        default_model=ollama_model_name
    )


class EchoType(BaseModel):
    text: str


echo_inference = Inference(
    id="echo",
    version=1,
    instructions="Return the provided EchoType object exactly as it was provided. Do not modify the text property.",
    input_type=EchoType,
    output_type=EchoType,
    metrics=[echo(),],
    examples=[
        (EchoType(text="Hello, world!"), EchoType(text="Hello, world!")),
        (EchoType(text="How are you?"), EchoType(text="How are you?")),
    ],
    eval_runs=10,
    threshold=0.7,
    temperature=0.0
)
