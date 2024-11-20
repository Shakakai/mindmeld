from mindmeld import Inference
from pydantic import BaseModel


class StringComparison(BaseModel):
    a: str
    b: str


class Delta(BaseModel):
    change_description: str


string_comparison_inference = Inference(
    id="echo",
    instructions="Describe the difference between the `a` and `b` strings.",
    input_type=StringComparison,
    output_type=Delta,
)
