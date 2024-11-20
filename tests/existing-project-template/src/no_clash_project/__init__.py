from mindmeld import Inference
from mindmeld.metrics import echo
from pydantic import BaseModel
from .sub import string_comparison_inference


class Echo(BaseModel):
    text: str


echo_inference = Inference(
    id="echo",
    instructions="Echo the input text",
    input_type=Echo,
    output_type=Echo,
    metrics=[echo()],
)
