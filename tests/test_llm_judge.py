from pydantic import BaseModel
from mindmeld.inference import Inference, run_inference
from mindmeld.eval import eval_inference
from mindmeld.metrics.llm_judge import llm_judge


class Person(BaseModel):
    name: str
    age: int


class BirthdayMessage(BaseModel):
    message: str


birthday_message_inference = Inference(
    id="birthday_message",
    version=1,
    instructions="Generate a birthday message for this person.",
    input_type=Person,
    output_type=BirthdayMessage,
    metrics=[
        llm_judge("Does this message sound like a birthday message?"),
        llm_judge("Is this message positive?")
    ],
    eval_threshold=0.8,
    examples=[
        (Person(name="Alice", age=30), BirthdayMessage(message="Happy 30th birthday, Alice!")),
        (Person(name="Bob", age=40), BirthdayMessage(message="Happy 40th birthday, Bob!")),
    ],
    eval_runs=5
)


def test_birthday_message_inference(ollama_runtime_config):
    test_person = Person(name="Alice", age=30)

    # Run the inference
    inference_result = run_inference(
        inference=birthday_message_inference,
        input_data=test_person,
        runtime_config=ollama_runtime_config
    )

    # Assert the result
    assert isinstance(inference_result.result, BirthdayMessage)


def test_birthday_message_eval(ollama_runtime_config):
    test_person = Person(name="Alice", age=30)

    # Run the inference
    eval_result = eval_inference(
        inference=birthday_message_inference,
        input_data=test_person,
        runtime_config=ollama_runtime_config
    )

    # Assert the result
    assert eval_result.success
