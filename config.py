"""
config.py

This configuration file establishes all system-wide parameters to be shared across modules,
such as DatasetLoader, Model, Trainer, Evaluation, etc. It centralizes settings for training,
dataset paths, model selection, evaluation metrics, logging, and database connectivity.

It first attempts to load configuration parameters from an external YAML file ("config.yaml").
If that file is not found or some keys are missing, default values are used.

The configuration structure is defined as follows:

1. training:
   - learning_rate, batch_size, epochs, optimizer: Parameters reserved for neural model training.
2. dataset:
   - s2orc_path: Local path for the S2ORC full-text dataset.
   - s2ag_path: Local path for the S2AG metadata.
   - sampling_threshold: Minimum percentage threshold (e.g., 3) for field filtering.
3. model:
   - type: Which citation recommendation model to use ("BM25", "NCN", "LCR", "Galactica").
   - hyperparameters: Reserved dictionary for model-specific hyperparameters.
4. evaluation:
   - recall_k: Top-k value for computing Recall@k.
   - mrr_k: Top-k value for computing MRR@k.
5. logging:
   - level: Logging level (e.g., "INFO", "DEBUG").
6. database:
   - mongodb_uri: URI for the MongoDB instance.
   - database_name: Name of the MongoDB database for storing benchmark data.

This configuration file is a central point for experiment reproduction and serves as the single source
of truth for parameter definitions across the entire project.
"""

import os
import logging
from typing import Any, Dict

try:
    import yaml
except ImportError as e:
    raise ImportError("Please install PyYAML package to use the config module: pip install PyYAML") from e

# Define default configuration parameters
DEFAULT_CONFIG: Dict[str, Any] = {
    "training": {
        "learning_rate": None,  # Not specified in the paper; to be tuned if required
        "batch_size": None,     # Not specified in the paper; to be tuned if required
        "epochs": None,         # Not specified in the paper; to be tuned if required
        "optimizer": None       # Not specified in the paper; to be tuned if required
    },
    "dataset": {
        "s2orc_path": "path/to/s2orc/dataset",  # Local path for S2ORC full-text dataset
        "s2ag_path": "path/to/s2ag/metadata",     # Local path for S2AG metadata
        "sampling_threshold": 3                   # Minimum percentage threshold (3%) for field filtering
    },
    "model": {
        "type": "BM25",         # Options: BM25, NCN, LCR, Galactica; BM25 is chosen as the baseline
        "hyperparameters": {}   # Reserved for model-specific hyperparameters (e.g., for NCN, LCR, Galactica)
    },
    "evaluation": {
        "recall_k": 10,  # Top 10 recommendations for Recall evaluation
        "mrr_k": 10      # Top 10 recommendations for MRR evaluation
    },
    "logging": {
        "level": "INFO"  # Logging level to control verbosity across modules
    },
    "database": {
        "mongodb_uri": "mongodb://localhost:27017",  # MongoDB connection URI
        "database_name": "citation_benchmark"          # Database name for storing benchmark data
    }
}


def _deep_update(source: Dict[Any, Any], overrides: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Recursively update the source dictionary with overrides.
    This helper ensures that nested dictionaries are merged correctly.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            source[key] = _deep_update(source[key], value)
        else:
            source[key] = value
    return source


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration parameters from an external YAML file.
    If the file is not found or a key is missing, default values are used.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the merged configuration parameters.
    """
    user_config: Dict[str, Any] = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    user_config = loaded
                else:
                    logging.warning("YAML configuration file is empty or improperly formatted; using defaults.")
        except Exception as e:
            logging.error("Error reading YAML config at '%s': %s. Using default configuration.", config_path, e)
    else:
        logging.info("Configuration file '%s' not found; using default configuration.", config_path)

    # Merge the default configuration with user loaded configuration
    merged_config = _deep_update(DEFAULT_CONFIG.copy(), user_config)
    return merged_config


# Load configuration at module level so that it is shared across modules.
config: Dict[str, Any] = load_config()

# Set up logging based on configuration
logging_level = getattr(logging, config["logging"].get("level", "INFO").upper(), logging.INFO)
logging.basicConfig(level=logging_level,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Expose the configuration dictionary as the module-level variable 'config'
if __name__ == "__main__":
    # For quick testing: print the loaded configuration
    import pprint
    pprint.pprint(config)
