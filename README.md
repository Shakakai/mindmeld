# mindmeld

A Python framework for testing and evaluating Large Language Model (LLM) content generation with a focus on maintaining consistent style and tone across responses.

## Features

- Define structured input/output schemas using Pydantic models
- Create inference configurations with example-based learning
- Evaluate LLM outputs using automated judges
- Set quality thresholds for acceptance criteria
- Support for multiple evaluation metrics and runs

## Usage Example

Here's an example of defining an inference configuration for generating Reddit comments in a specific user's style:

```python
from mindmeld.inference import Inference, RuntimeConfig, run_inference, eval_inference, AIProvider, AIModel
from mindmeld.metrics.echo import echo
from pydantic import BaseModel
import os
# Define your input/output schemas
class EchoInput(BaseModel):
    text: str

class EchoOutput(BaseModel):
    result: str

# Register the active models and set defaults
runtime_config =RuntimeConfig(
    models=[
        AIModel(
            provider=AIProvider(name="openai"),
            name="gpt-4o"
        )
    ],
    eval_model="gpt-4o",
    default_model="gpt-4o"
)

# Create an inference configuration
inference = Inference(
    id="echo",
    instructions="""
    Echo back the input text.
    """,
    input_type=EchoInput,
    output_type=EchoOutput,
    examples=[
        (EchoInput(text="Hello"), EchoOutput(result="Hello")),
        (EchoInput(text="Test"), EchoOutput(result="Test"))
    ],
    metrics=[
        echo()
    ],
    eval_runs=3,
    eval_threshold=0.9
)

# Run inference and evaluation
input_data = EchoInput(text="Hello, world!")
result = run_inference(inference, input_data, runtime_config)
eval_result = eval_inference(inference, input_data, runtime_config)
```

## Installation

[Add installation instructions once package is published]

## Documentation

[Add link to detailed documentation once available]

## License

MIT License. See the LICENSE file for details.