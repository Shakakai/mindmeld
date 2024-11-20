import json
import os
import shutil
from pathlib import Path
import pytest
from click.testing import CliRunner
from mindmeld.runner import cli, find_mindmeld_config, load_runtime_config, find_inference, find_all_inference

CURRENT_FILEPATH = os.path.abspath(__file__)
TEST_DIR = Path(CURRENT_FILEPATH).parent
PROJECT_DIR = TEST_DIR / "example-project"
PROJECT_NAME = "no_clash_project"


def setup_project(folder: str):
    """Create and clean up example project directory for tests"""
    template_dir = TEST_DIR / folder
    if not template_dir.exists():
        raise Exception("Template directory not found")

    if PROJECT_DIR.exists():
        shutil.rmtree(PROJECT_DIR)

    # Copy template directory contents to project directory
    shutil.copytree(template_dir, PROJECT_DIR)

    # Change to project directory
    orig_dir = os.getcwd()
    os.chdir(PROJECT_DIR)
    return orig_dir


def cleanup_project(orig_dir: str):
    os.chdir(orig_dir)
    shutil.rmtree(PROJECT_DIR)


@pytest.fixture
def new_example_project():
    """Create and clean up example project directory for tests"""
    folder = "new-project-template"
    cwd = setup_project(folder)
    yield PROJECT_DIR
    cleanup_project(cwd)


@pytest.fixture
def existing_example_project():
    """Create and clean up example project directory for tests"""
    folder = "existing-project-template"
    cwd = setup_project(folder)
    yield PROJECT_DIR
    cleanup_project(cwd)


@pytest.fixture
def runtime_config_project(existing_example_project):
    os.environ["MINDMELD_CONFIG"] = str(PROJECT_DIR / "mindmeld.json")
    config_path = find_mindmeld_config()
    runtime_config = load_runtime_config(config_path)
    return runtime_config


@pytest.mark.skip("failing right now")
def test_init_command(new_example_project):
    runner = CliRunner()

    # Run init command with input
    result = runner.invoke(cli, ["init"], input=f"y\n{PROJECT_NAME}\n")

    assert result.exit_code == 0
    assert Path("datasets").exists()
    assert Path("inference_configs").exists()
    assert Path("mindmeld.json").exists()

    # Verify config contents
    with open("mindmeld.json") as f:
        config = json.load(f)
        assert config["root_module"] == PROJECT_NAME
        assert config["dataset_dir"] == "datasets"
        assert config["inference_config_dir"] == "inference_configs"


@pytest.mark.skip("failing right now")
def test_init_command_abort(new_example_project):
    runner = CliRunner()

    # Run init command but abort
    result = runner.invoke(cli, ["init"], input="n\n")

    assert result.exit_code == 0
    assert "Aborted." in result.output
    assert not Path("datasets").exists()
    assert not Path("inference_configs").exists()


def test_inference_command_no_config(new_example_project):
    runner = CliRunner()
    result = runner.invoke(cli, ["inference", "test-inference"])

    assert result.exit_code != 0
    assert "No mindmeld.json configuration file found" in result.output


def test_generate_dataset_command_no_config(new_example_project):
    runner = CliRunner()
    result = runner.invoke(cli, ["generate-dataset", "test-inference"])

    assert result.exit_code != 0
    assert "No mindmeld.json configuration file found" in result.output


def test_find_module(runtime_config_project):
    echo_inference = find_inference(runtime_config_project, "echo")
    assert echo_inference is not None


def test_find_all_modules(runtime_config_project):
    all_inferences = find_all_inference(runtime_config_project)
    assert len(all_inferences) == 2


def test_generate_dataset_command(existing_example_project):
    runner = CliRunner()
    result = runner.invoke(cli, ["generate-dataset", "echo"])

    assert result.exit_code == 0
    assert Path("datasets/echo.json").exists()


def test_eval_command(existing_example_project):
    runner = CliRunner()
    result = runner.invoke(cli, ["eval", "echo"])

    assert result.exit_code == 0
    assert "Evaluation complete" in result.output


def test_inference_command(existing_example_project):
    runner = CliRunner()
    result = runner.invoke(cli, ["inference", "echo"], input="{\"text\": \"Hello\"}\n")

    assert result.exit_code == 0
    assert "Hello" in result.output


def test_optimize_command(existing_example_project):
    runner = CliRunner()
    result = runner.invoke(cli, ["optimize", "echo", "--max-iterations", "1"])

    assert result.exit_code == 0
    assert "Optimization complete" in result.output
