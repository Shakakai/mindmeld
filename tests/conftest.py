import pytest
from pydantic import BaseModel

from mindmeld.inference import RuntimeConfig, AIModel, AIProvider, Inference


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

