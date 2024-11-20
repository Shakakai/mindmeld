import uuid
from types import ModuleType

import click
import json
import sys
import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Optional, List
import os

from mindmeld import generate_synthetic_pairs, eval_inference, optimize_inference
from mindmeld.eval import eval_dataset_inference
from mindmeld.inference import RuntimeConfig, Inference, run_inference, DataEntry, AIModel, AIProvider, Dataset, \
    InferenceConfig


def find_mindmeld_config() -> Optional[Path]:
    """
    Look for a .mindmeld configuration file first in MINDMELD_CONFIG env var,
    then in the current directory or its parents.
    
    Returns:
        Optional[Path]: Path to the mindmeld.json file if found, None otherwise
    """
    # Check environment variable first
    env_config = os.environ.get('MINDMELD_CONFIG')
    if env_config:
        config_path = Path(env_config)
        if config_path.exists():
            return config_path

    # Fall back to searching directory tree
    current = Path.cwd()
    while current != current.parent:
        config_path = current / 'mindmeld.json'
        if config_path.exists():
            return config_path
        current = current.parent
    return None


def load_runtime_config(config_path: Path) -> RuntimeConfig:
    """
    Load the RuntimeConfig from a mindmeld.json file
    
    Args:
        config_path: Path to the mindmeld.json file
        
    Returns:
        RuntimeConfig: The loaded configuration
        
    Raises:
        click.ClickException: If the config file is invalid
    """
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
        cfg = RuntimeConfig(**config_data)
        cfg.resolve_paths(config_path)
        return cfg
    except Exception as e:
        raise click.ClickException(f"Failed to load mindmeld.json config: {str(e)}")


def read_input_data(input_file: Optional[str] = None) -> str:
    """
    Read input data from a file, stdin, or interactive prompt
    
    Args:
        input_file: Optional path to input JSON file
        
    Returns:
        str: The input data as a string
        
    Raises:
        click.ClickException: If the input file cannot be read
    """
    if input_file:
        try:
            with open(input_file, 'r') as f:
                return f.read().strip()
        except Exception as e:
            raise click.ClickException(f"Failed to read input file {input_file}: {str(e)}")
            
    # Check if we have data piped in
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    else:
        # Interactive mode - prompt for input
        return click.prompt("Enter input data", type=str)


def import_module(module_name: str) -> Optional[ModuleType]:
    """
    Import a module by name, handling import errors gracefully.

    Args:
        module_name: Fully qualified module name to import

    Returns:
        Optional[ModuleType]: The imported module if successful, None otherwise
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        return None


def find_inference_in_module(
        module_name: str,
        inference_id: str
) -> Optional[Inference]:
    """
    Search recursively through a module and its submodules for an Inference object with matching id.
    
    Args:
        module_name:
        module:
        module_path: Fully qualified module path to search
        inference_id: ID of the inference to find
        
    Returns:
        Optional[Inference]: The matching inference object if found, None otherwise
    """
    module = import_module(module_name)

    # Search for inference objects in current module
    for name, obj in inspect.getmembers(module):
        if isinstance(obj, Inference) and obj.id == inference_id:
            return obj

    # Recursively search submodules
    if hasattr(module, '__path__'):
        for _, submodule_name, is_pkg in pkgutil.iter_modules(module.__path__):
            full_submodule_path = f"{module_name}.{submodule_name}"
            result = find_inference_in_module(full_submodule_path, inference_id)
            if result:
                return result
                
    return None


def find_inference(runtime_config: RuntimeConfig, inference_id: str) -> Inference:
    """
    Find an Inference object with the given ID in the project's root module.
    
    Args:
        runtime_config: The project's runtime configuration
        inference_id: ID of the inference to find
        
    Returns:
        Inference: The matching inference object
        
    Raises:
        click.ClickException: If no matching inference is found
    """
    module = import_module(runtime_config.root_module)
    if module is None:
        for python_source in runtime_config.get_python_sources():
            sys.path.insert(0, str(python_source))
        module = import_module(runtime_config.root_module)

    if module is None:
        click.echo(f"Warning: Failed to import module {runtime_config.root_module}", err=True)
        return None

    inference = find_inference_in_module(runtime_config.root_module, inference_id)
    if not inference:
        raise click.ClickException(
            f"No inference found with id '{inference_id}' in module '{runtime_config.root_module}'"
        )
    return inference


def find_all_inference(runtime_config: RuntimeConfig) -> List[Inference]:
    """
    Find all Inference objects in the project's root module.

    Args:
        runtime_config: The project's runtime configuration
    """
    module = import_module(runtime_config.root_module)
    if module is None:
        for python_source in runtime_config.get_python_sources():
            sys.path.insert(0, str(python_source))
        module = import_module(runtime_config.root_module)

    if module is None:
        click.echo(f"Warning: Failed to import module {runtime_config.root_module}", err=True)
        return []

    inferences = []
    for name, obj in inspect.getmembers(module):
        if isinstance(obj, Inference):
            inferences.append(obj)
    return inferences


@click.group()
def cli():
    """MindMeld CLI tool for managing AI inference projects"""
    pass


@cli.command()
@click.argument('inference_id')
@click.option('--input', '-i', 'input_file', type=click.Path(exists=True, dir_okay=False, path_type=str),
              help="JSON file containing input data. If not provided, will read from stdin or prompt.")
def inference(inference_id: str, input_file: Optional[str] = None):
    """
    Run inference using the given input data and runtime configuration found in the .mindmeld file.
    
    Input data can be provided in three ways:
    1. Input file: mindmeld inference my-inference --input data.json
    2. Piped input: cat data.json | mindmeld inference my-inference
    3. Interactive prompt: mindmeld inference my-inference
    """
    # Find and load config
    config_path = find_mindmeld_config()
    if not config_path:
        raise click.ClickException("No mindmeld.json configuration file found in current directory or its parents")
    
    runtime_config = load_runtime_config(config_path)
    
    # Read input data
    try:
        input_data = read_input_data(input_file)
        input_json = json.loads(input_data)
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON input: {str(e)}")

    # Run inference
    try:
        inference = find_inference(runtime_config, inference_id)
        result = run_inference(
            inference=inference,
            input_data=inference.input_type(**input_json),
            runtime_config=runtime_config
        )
        
        # Output result as JSON
        output = result.model_dump()
        click.echo(json.dumps(output, indent=2))
        
    except Exception as e:
        raise click.ClickException(f"Inference failed: {str(e)}")


@cli.command()
@click.argument('inference_id')
def eval(inference_id: str):
    """
    Evaluate an inference using stored test data from the dataset directory.
    
    Loads the dataset for the specified inference, runs inference on each input,
    and compares against expected outputs to generate evaluation metrics.

    Args:
        inference_id (str): The ID of the inference to evaluate
    """
    # Find and load config
    config_path = find_mindmeld_config()
    if not config_path:
        raise click.ClickException("No mindmeld.json configuration file found in current directory or its parents")
    
    runtime_config = load_runtime_config(config_path)
    
    # Find the inference
    inference = find_inference(runtime_config, inference_id)
    if not inference:
        raise ValueError(f"No inference found with id '{inference_id}'")

    # Load dataset
    try:
        dataset = Dataset.load(runtime_config, inference)
    except FileNotFoundError:
        raise click.ClickException(f"No dataset found for inference '{inference_id}'")

    eval_result = eval_dataset_inference(inference, dataset, runtime_config)

    click.echo(f"Evaluation complete for {inference_id}: {eval_result.success}")
    click.echo(f"Eval Result: {eval_result}")


@cli.command()
@click.argument('inference_id')
@click.option('--max-iterations', 'max_iterations', type=int)
def optimize(inference_id: str, max_iterations: int = 3):
    """
    Optimize the inference configuration for better performance.

    This command will run a series of optimizations on the inference configuration to improve performance.
    The optimizations may include model fine-tuning, hyperparameter tuning, or other techniques.

    Args:
        max_iterations:
        inference_id (str): The ID of the inference to optimize
    """
    config_path = find_mindmeld_config()
    if not config_path:
        raise click.ClickException("No mindmeld.json configuration file found in current directory or its parents")

    runtime_config = load_runtime_config(config_path)

    # Find the inference
    inference = find_inference(runtime_config, inference_id)
    if not inference:
        raise ValueError(f"No inference found with id '{inference_id}'")

    dataset = Dataset.load(runtime_config, inference)
    optimize_result = optimize_inference(inference, dataset, runtime_config, max_iterations=max_iterations)
    if optimize_result.success:
        ic = inference.get_config(runtime_config)
        ic.prompt = optimize_result.best_prompt
        InferenceConfig.save(runtime_config, inference, ic)
        click.echo(f"Optimization complete for {inference_id}: {optimize_result.success}")
        click.echo(f"Best prompt: {optimize_result.best_prompt}")
        click.echo(f"Improvement: {optimize_result.improvement}")
    else:
        click.echo(f"Optimization failed for {inference_id}")


@cli.command()
@click.argument('inference_id')
def generate_dataset(inference_id: str):
    """
    Generate a synthetic dataset for the specified inference and save it to the configured dataset directory.
    
    Uses the generate_synthetic_pairs function to create input/output pairs, then stores them as DataEntry 
    objects in a JSON file within the runtime_config's dataset directory.

    Args:
        inference_id (str): The ID of the inference to generate data for
    """
    # Find and load config
    config_path = find_mindmeld_config()
    if not config_path:
        raise click.ClickException("No mindmeld.json configuration file found in current directory or its parents")
    
    runtime_config = load_runtime_config(config_path)
    
    # Find the inference
    inference = find_inference(runtime_config, inference_id)
    if not inference:
        raise ValueError(f"No inference found with id '{inference_id}'")
    
    # Generate synthetic input/output pairs
    pairs = generate_synthetic_pairs(runtime_config, inference)
    
    # Create dataset entries
    dataset = Dataset()
    for input_data, output_data in pairs:
        entry = DataEntry(
            id=str(uuid.uuid4()),
            input=input_data,
            expected=output_data
        )
        dataset.entries.append(entry)
    # Save to dataset directory
    Dataset.save(runtime_config, inference, dataset)


@cli.command()
def init():
    """Initialize a new mindmeld project in the current directory."""
    print(f"Current directory: {Path.cwd()}")
    pdm_cwd = os.getenv("PDM_RUN_CWD")
    current_dir = Path.cwd() if pdm_cwd is None else Path(pdm_cwd)
    print(f"Resolved Current directory: {current_dir}")

    # Ask for permission to create directories
    dataset_dir = current_dir / "datasets"
    inference_dir = current_dir / "inference_configs"
    
    if not click.confirm(f"This will create directories '{dataset_dir}' and '{inference_dir}' in the current directory. Continue?"):
        click.echo("Aborted.")
        return
        
    # Create directories
    dataset_dir.mkdir(exist_ok=True)
    inference_dir.mkdir(exist_ok=True)

    # Get root module name
    root_module = click.prompt("What is the root Python module name for your project?", type=str)
    python_source = click.prompt("Where are your Python source files located?", type=str)
    
    # Create default config
    runtime_config = RuntimeConfig(
        models=[
            AIModel(
                provider=AIProvider(name="openai"),
                name="gpt-4o"
            )
        ],
        eval_model="gpt-4o",
        default_model="gpt-4o",
        root_module=root_module,
        project_root=str(Path.cwd()),
        dataset_dir="datasets",
        inference_config_dir="inference_configs",
        python_sources=[python_source]
    )

    # Save config file
    mindmeld_filepath = current_dir / "mindmeld.json"
    with open(mindmeld_filepath, "w") as f:
        json.dump(runtime_config.model_dump(), f, indent=2)


if __name__ == '__main__':
    cli()
