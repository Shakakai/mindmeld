from mindmeld.synthetic import generate_synthetic_input, generate_synthetic_pairs
from mindmeld.inference import Inference
from pydantic import BaseModel, Field
from typing import Optional


class Person(BaseModel):
    name: str
    age: int
    occupation: str
    education_level: Optional[str] = None
    location: str


class PoliticalViews(BaseModel):
    economic_stance: str  # e.g. "conservative", "liberal", "moderate"
    social_stance: str
    party_affiliation: Optional[str]
    key_issues: list[str]
    voting_likelihood: float = Field(
        description="The percent likelihood that the person will vote in the next election. A value between 0 and 1.",
        gte=0.0,
        lte=1.0
    )


def test_generate_synthetic_input(runtime_config):
    # Create test inference
    test_inference = Inference(
        id="political_views_inference",
        instructions="Based on the person's demographic information, "
                     "predict their most likely political views and stances on key issues",
        input_type=Person,
        output_type=PoliticalViews
    )

    # Generate synthetic input data
    results = generate_synthetic_input(
        runtime_config=runtime_config,
        inference=test_inference,
        count=3
    )

    # Verify we got the requested number of results
    assert len(results) == 3

    # Verify each result is of the correct type and has valid data
    for result in results:
        assert isinstance(result, Person)
        assert isinstance(result.name, str)
        assert len(result.name) > 0
        assert isinstance(result.age, int)
        assert result.age >= 18  # Voting age
        assert isinstance(result.occupation, str)
        assert isinstance(result.location, str)
        
        # Education is optional but should be a string if present
        if result.education_level is not None:
            assert isinstance(result.education_level, str)

    # Verify results are unique
    names = [r.name for r in results]
    assert len(set(names)) == len(names), "Generated persons should be unique"


def test_generate_synthetic_pairs(runtime_config):
    # Create test inference
    test_inference = Inference(
        id="political_views_inference",
        instructions="Based on the person's demographic information, predict their most likely political views and "
                     "stances on key issues. Consider factors like age, occupation, education, "
                     "and location to inform the prediction.",
        input_type=Person,
        output_type=PoliticalViews
    )

    # Generate synthetic input/output pairs
    pairs = generate_synthetic_pairs(
        runtime_config=runtime_config,
        inference=test_inference,
        count=3
    )

    # Verify we got the requested number of pairs
    assert len(pairs) == 3

    # Verify each pair has valid input and output
    for input_data, output_data in pairs:
        # Check input (Person)
        assert isinstance(input_data, Person)
        assert isinstance(input_data.name, str)
        assert isinstance(input_data.age, int)
        assert isinstance(input_data.occupation, str)
        assert isinstance(input_data.location, str)
        assert input_data.age >= 18

        # Check output (PoliticalViews)
        assert isinstance(output_data, PoliticalViews)
        assert isinstance(output_data.economic_stance, str)
        assert isinstance(output_data.social_stance, str)
        assert isinstance(output_data.key_issues, list)
        assert isinstance(output_data.voting_likelihood, float)
        assert 0 <= output_data.voting_likelihood <= 1

        # If party affiliation is present, it should be a string
        if output_data.party_affiliation is not None:
            assert isinstance(output_data.party_affiliation, str)

        # Verify both input and output have valid data
        assert len(input_data.name) > 0
        assert len(output_data.economic_stance) > 0
        assert len(output_data.social_stance) > 0
        assert len(output_data.key_issues) > 0
