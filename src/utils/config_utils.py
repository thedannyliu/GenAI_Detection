import yaml
import importlib
from typing import Any, Dict

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: The configuration dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        raise

def dynamic_import(module_path: str, class_name: str) -> Any:
    """Dynamically imports a class from a given module path.

    Args:
        module_path (str): The full path to the module (e.g., "src.models.vlm.clip_model").
        class_name (str): The name of the class to import (e.g., "CLIPModelWrapper").

    Returns:
        Any: The imported class.
    
    Raises:
        ImportError: If the module or class cannot be imported.
    """
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        print(f"Error importing {class_name} from {module_path}: {e}")
        raise
    except AttributeError:
        print(f"Error: Class {class_name} not found in module {module_path}")
        raise

def get_instance_from_config(config: Dict[str, Any], **kwargs) -> Any:
    """Creates an instance of a class specified in the config.

    The config dictionary must contain:
    - 'module': The full path to the module (e.g., "src.models.vlm.clip_model")
    - 'class': The name of the class (e.g., "CLIPModelWrapper")
    - 'params' (optional): A dictionary of parameters to pass to the class constructor.

    Args:
        config (Dict[str, Any]): The configuration dictionary for the instance.
        **kwargs: Additional keyword arguments to pass to the class constructor,
                  these will be merged with 'params' from the config, with kwargs taking precedence.

    Returns:
        Any: An instance of the specified class.
    """
    if not all(k in config for k in ['module', 'class']):
        raise ValueError("Configuration for instance must contain 'module' and 'class' keys.")

    module_path = config['module']
    class_name = config['class']
    constructor_params = config.get('params', {})
    
    # Merge config params with kwargs, kwargs take precedence
    # constructor_params already contains config.get('params', {})
    # kwargs are additional arguments passed to get_instance_from_config
    # The final set of parameters to be passed to Klass constructor should be a merge of these.
    
    final_constructor_args = constructor_params.copy() # Start with params from the config file
    final_constructor_args.update(kwargs) # Override/add with any kwargs passed directly to get_instance_from_config

    Klass = dynamic_import(module_path, class_name)
    return Klass(**final_constructor_args) 