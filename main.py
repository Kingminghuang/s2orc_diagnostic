"""
main.py

This module orchestrates the end-to-end workflow for the citation recommendation benchmark experiment.
It performs the following steps:

1. Loads configuration parameters from config.yaml via config.py.
2. Initializes logging.
3. Loads and preprocesses the S2ORC and S2AG datasets via DatasetLoader.
4. Constructs the diagnostic dataset from preprocessed sentences.
5. Instantiates the appropriate citation recommendation model based on the configuration.
   - For BM25 (the baseline), it builds a candidate corpus from tokenized diagnostic sentences.
   - For neural models (e.g., NCN, LCR, Galactica), it initializes the NeuralModel and invokes Trainer to run training.
6. Evaluates the model on the diagnostic dataset to compute Recall@10 and MRR@10.
7. Logs and prints the evaluation outcomes.

This integration layer strictly follows the design, interfaces, and configuration provided.
"""

import sys
import logging

# Import configuration, dataset loader, model, trainer, and evaluation modules.
from config import config
from dataset_loader import DatasetLoader, CitingSentence
from model import BM25Model, NeuralModel, Model
from trainer import Trainer
from evaluation import Evaluation

def main() -> None:
    # Initialize main logger using configuration logging settings.
    logger = logging.getLogger("main")
    logger.setLevel(logging.getLevelName(config["logging"].get("level", "INFO").upper()))
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)

    logger.info("Starting citation recommendation benchmark experiment.")

    try:
        # Step 1: Data Loading and Preprocessing
        logger.info("Initializing DatasetLoader with S2ORC path: '%s' and S2AG path: '%s'.",
                    config["dataset"].get("s2orc_path", "path/to/s2orc/dataset"),
                    config["dataset"].get("s2ag_path", "path/to/s2ag/metadata"))
        dataset_loader = DatasetLoader(config)
        logger.info("Loading and preprocessing data...")
        diagnostic_dataset = dataset_loader.load_data()
        num_samples = len(diagnostic_dataset)
        logger.info("Loaded %d diagnostic sentences.", num_samples)
        if num_samples == 0:
            logger.error("No diagnostic sentences were loaded. Exiting experiment.")
            sys.exit(1)

        # Step 2: Construct candidate corpus for BM25 (list of tokenized sentence texts)
        # Here, each candidate is the tokenized version of the citing sentence text.
        candidate_corpus = [sentence.text.split() for sentence in diagnostic_dataset]
        logger.info("Constructed candidate corpus with %d tokenized documents.", len(candidate_corpus))

        # Step 3: Model Initialization based on configuration.
        model_type: str = config["model"].get("type", "BM25").upper()
        logger.info("Model type selected from configuration: %s", model_type)
        if model_type == "BM25":
            logger.info("Initializing BM25Model as baseline.")
            model_instance: Model = BM25Model(corpus=candidate_corpus,
                                              top_k=config["evaluation"].get("recall_k", 10))
        else:
            logger.info("Initializing NeuralModel with hyperparameters: %s",
                        config["model"].get("hyperparameters", {}))
            model_instance = NeuralModel(params=config["model"].get("hyperparameters", {}))
            # Step 4: Optional Training Phase (for neural models only)
            logger.info("Initializing Trainer for the neural model training phase.")
            trainer = Trainer(model=model_instance, training_dataset=diagnostic_dataset)
            trainer.train()

        # Step 5: Evaluation Phase - Compute Recall@10 and MRR@10
        logger.info("Initializing Evaluation with top-k: %d", config["evaluation"].get("recall_k", 10))
        evaluator = Evaluation(model=model_instance, dataset=diagnostic_dataset)
        evaluation_metrics = evaluator.evaluate()

        # Log and print final evaluation results
        logger.info("Evaluation complete. Metrics: %s", evaluation_metrics)
        print("Evaluation Metrics:")
        for metric_name, metric_value in evaluation_metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

    except Exception as err:
        logger.exception("An error occurred during the experiment: %s", err)
        sys.exit(1)

if __name__ == "__main__":
    main()
