from mindmeld.inference import Inference, MetricCallableType, InferenceType, run_inference, RuntimeConfig
from pydantic import BaseModel, Field
from typing import List


class StatementVerificationInput(BaseModel):
    context: BaseModel = Field(description="The original context or source information")
    statement: str = Field(description="A single statement to be verified")


class StatementVerificationOutput(BaseModel):
    is_faithful: bool = Field(description="Whether the statement can be inferred from the context")
    reasoning: str = Field(description="Explanation for the verification decision")


statement_verification_inference = Inference(
    id="statement_verification",
    instructions="""
    Given a context and a statement, determine if the statement can be faithfully inferred from the context.
    Provide a boolean result and explain your reasoning.
    """,
    input_type=StatementVerificationInput,
    output_type=StatementVerificationOutput
)


class StatementExtractionOutput(BaseModel):
    statements: List[str] = Field(description="List of extracted statements")


statement_extraction_inference = Inference(
    id="statement_extraction",
    instructions="""
    Given a text, break it down into individual statements.
    Each statement should be a single, coherent piece of information.
    If there are no obvious statements of information, return an empty list.
    Otherwise, return the list of extracted statements.
    """,
    input_type=BaseModel,
    output_type=StatementExtractionOutput
)


def faithfulness() -> MetricCallableType:
    """
    A metric that evaluates the faithfulness of an answer to a given context.

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
        # Step 1: Extract statements from the generated answer
        extraction_output = run_inference(statement_extraction_inference, output_data, runtime_config, test=True)
        
        statements = extraction_output.statements
        
        # Step 2: Verify each statement
        faithful_statements = 0
        for statement in statements:
            verification_input = StatementVerificationInput(
                context=input_data,
                statement=statement
            )
            verification_output = run_inference(
                statement_verification_inference,
                verification_input,
                runtime_config,
                test=True
            )
            
            if verification_output.is_faithful:
                faithful_statements += 1
        
        # Step 3: Calculate faithfulness score
        faithfulness_score = faithful_statements / len(statements) if len(statements) > 0 else 0.0
        
        return faithfulness_score

    __impl__.__name__ = "faithfulness"
    return __impl__


