from mindmeld.inference import Inference, MetricCallableType, InferenceType, run_inference, RuntimeConfig
from pydantic import BaseModel, Field
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from mindmeld.pydantic_utils import pydantic_to_vs


class QuestionGenerationInput(BaseModel):
    answer: BaseModel = Field(description="The answer from which to generate questions")
    num_questions: int = Field(description="The number of questions to generate")


class QuestionGenerationOutput(BaseModel):
    generated_questions: List[str] = Field(description="List of generated questions")


question_generation_inference = Inference(
    id="question_generation",
    version=1,
    instructions="""
    Given an answer data structure, generate a specified number of possible questions that this answer could be responding to.
    Ensure the generated questions are diverse and capture different aspects of the answer.
    """,
    input_type=QuestionGenerationInput,
    output_type=QuestionGenerationOutput
)


# Initialize the sentence transformer model
sentence_model = SentenceTransformer(
    'all-MiniLM-L6-v2',
    tokenizer_kwargs={'cleaup_tokenization_spaces': True}
)


def calculate_cosine_similarity(sentence1: str, sentence2: str) -> float:
    embeddings = sentence_model.encode([sentence1, sentence2])
    return np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))


def answer_similarity() -> MetricCallableType:
    """
    A metric that evaluates the relevance of an answer to a given question using question generation and similarity comparison.

    Args:
        runtime_config (RuntimeConfig): Configuration for the runtime environment.
        model_name (str): Name of the model to use for evaluation.

    Returns:
        Callable: A function that takes an Inference, system prompt, input data,
                  and output data, and returns a float score between 0 and 1.
    """

    def __impl__(
            runtime_config: RuntimeConfig,
            inference: Inference,
            system_prompt: str,
            input_data: BaseModel,
            output_data: BaseModel
    ) -> float:
        # Generate questions from the answer
        gen_input = QuestionGenerationInput(
           answer=output_data,
           num_questions=3  # You can adjust this number
        )
        gen_output = run_inference(question_generation_inference, gen_input, runtime_config, test=True)

        # Calculate similarities
        input_text = pydantic_to_vs(input_data)
        similarities = [calculate_cosine_similarity(input_text, q) for q in gen_output.generated_questions]
        mean_similarity = np.mean(similarities)

        return float(mean_similarity)

    __impl__.__name__ = "answer_similarity"
    return __impl__

