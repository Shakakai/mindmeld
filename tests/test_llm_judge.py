from pydantic import BaseModel
from mindmeld.inference import Inference, run_inference, DataEntry, Dataset, InferenceConfig
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
    examples=Dataset(entries=[
        DataEntry(
            input=Person(name="Alice", age=30),
            expected=BirthdayMessage(message="Happy 30th birthday, Alice!")
        ),
        DataEntry(
            input=Person(name="Bob", age=40),
            expected=BirthdayMessage(message="Happy 40th birthday, Bob!")
        ),
    ]),
    config=InferenceConfig(
        eval_runs=5,
        eval_threshold=0.8
    )
)


def test_birthday_message_inference(runtime_config):
    test_person = Person(name="Alice", age=30)

    # Run the inference
    inference_result = run_inference(
        inference=birthday_message_inference,
        input_data=test_person,
        runtime_config=runtime_config
    )

    # Assert the result
    assert isinstance(inference_result.result, BirthdayMessage)


def test_birthday_message_eval(runtime_config):
    test_person = Person(name="Alice", age=30)

    # Run the inference
    eval_result = eval_inference(
        inference=birthday_message_inference,
        input_data=test_person,
        runtime_config=runtime_config
    )

    # Assert the result
    assert eval_result.success
