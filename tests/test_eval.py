import pytest

from mindmeld.eval import eval_inference
from mindmeld.inference import Inference, RuntimeConfig, AIProvider, AIModel, run_inference
from pydantic import BaseModel

from mindmeld.metrics.echo import echo
from mindmeld.metrics.llm_judge import llm_judge


class EchoInput(BaseModel):
    text: str


class EchoOutput(BaseModel):
    result: str


echo_inference = Inference(
    id="echo",
    version=1,
    instructions="Echo the input text",
    input_type=EchoInput,
    output_type=EchoOutput,
    metrics=[echo(),],
    examples=[
        (EchoInput(text="Hello, world!"), EchoOutput(result="Hello, world!")),
        (EchoInput(text="How are you?"), EchoOutput(result="How are you?")),
    ]
)


def test_echo_inference(runtime_config):
    test_text = "Hello, world!"

    # Create test input
    echo_input = EchoInput(text=test_text)

    # Run the inference
    inference_result = run_inference(
        inference=echo_inference,
        input_data=echo_input,
        runtime_config=runtime_config
    )

    # Assert the result
    assert isinstance(inference_result.result, EchoOutput)
    assert inference_result.result.result == test_text


def test_echo_eval(runtime_config):
    test_text = "Hello, world!"

    # Create test input
    echo_input = EchoInput(text=test_text)

    # Run the inference
    eval_result = eval_inference(
        inference=echo_inference,
        input_data=echo_input,
        runtime_config=runtime_config
    )

    # Assert the result
    assert eval_result.success


class Person(BaseModel):
    name: str
    age: int


class BirthdayMessage(BaseModel):
    message: str


birthday_message_inference = Inference(
    id="birthday_message",
    version=1,
    instructions="Generate a birthday message",
    input_type=Person,
    output_type=BirthdayMessage,
    metrics=[
        llm_judge("Does this message sound like a birthday message?"),
        llm_judge("Is this message positive?")
    ],
    examples=[
        (Person(name="Alice", age=30), BirthdayMessage(message="Happy 30th birthday, Alice!")),
        (Person(name="Bob", age=40), BirthdayMessage(message="Happy 40th birthday, Bob!")),
    ],
    eval_runs=5
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
