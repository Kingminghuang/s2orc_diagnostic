"""
evaluation.py

This module implements the Evaluation class to compute evaluation metrics for the citation recommendation models.
It processes a diagnostic dataset (a list of CitingSentence objects) and computes Recall@10 and MRR@10 based on the model's predictions.
The Evaluation class relies on helper functions compute_recall and compute_mrr to aggregate per-sample metrics.
Configuration parameters such as the number of top candidates (k) are sourced from config.yaml.

The Evaluation class strictly follows the design specifications and is intended to be integrated into the overall benchmarking pipeline.
"""

import logging
import numpy as np
from typing import List, Dict, Any

# Import configuration from config.py
from config import config

# Import the CitingSentence class definition; 
# Avoid circular imports by not importing other modules from this project that use Evaluation.
# Assuming dataset_loader.CitingSentence is accessible:
try:
    from dataset_loader import CitingSentence
except ImportError as e:
    raise ImportError("Failed to import CitingSentence from dataset_loader. Please ensure module structure is correct.") from e

# Set up logger based on configuration settings
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, config["logging"].get("level", "INFO").upper(), logging.INFO))


class Evaluation:
    """
    Evaluation class to compute Recall@10 and MRR@10 for citation recommendation models.
    
    Attributes:
        model: An instance of a subclass of Model, implementing forward() and retrieve_candidates().
        dataset (List[CitingSentence]): List of preprocessed CitingSentence objects.
        recall_k (int): The top-k value for recall computation.
        mrr_k (int): The top-k value for MRR computation.
    """
    def __init__(self, model: Any, dataset: List[CitingSentence]) -> None:
        """
        Initialize the Evaluation module with a citation recommendation model and a diagnostic dataset.
        
        Args:
            model: A citation recommendation model implementing the Model interface.
            dataset (List[CitingSentence]): List of CitingSentence objects.
        """
        self.model = model
        self.dataset = dataset
        # Retrieve evaluation parameters from configuration (default to 10 if not specified)
        self.recall_k: int = int(config["evaluation"].get("recall_k", 10))
        self.mrr_k: int = int(config["evaluation"].get("mrr_k", 10))
        logger.info("Evaluation initialized with recall_k=%d and mrr_k=%d", self.recall_k, self.mrr_k)

    @staticmethod
    def compute_recall(predictions: List[str], ground_truth: str, k: int) -> float:
        """
        Compute recall for a single sample: 1.0 if ground_truth exists in top k predictions, else 0.0.
        
        Args:
            predictions (List[str]): Ranked candidate citation identifiers.
            ground_truth (str): The expected citation candidate identifier.
            k (int): The number of top predictions to consider.
            
        Returns:
            float: Recall value (1.0 or 0.0).
        """
        top_predictions = predictions[:k]
        recall_value = 1.0 if ground_truth in top_predictions else 0.0
        return recall_value

    @staticmethod
    def compute_mrr(predictions: List[str], ground_truth: str, k: int) -> float:
        """
        Compute Mean Reciprocal Rank (MRR) for a single sample.
        Returns 1/(rank_index+1) if ground_truth is found within the top k predictions; otherwise, 0.0.
        
        Args:
            predictions (List[str]): Ranked candidate citation identifiers.
            ground_truth (str): The expected citation candidate identifier.
            k (int): The number of top predictions to consider.
            
        Returns:
            float: Reciprocal rank value.
        """
        top_predictions = predictions[:k]
        for idx, candidate in enumerate(top_predictions):
            if candidate == ground_truth:
                return 1.0 / (idx + 1)
        return 0.0

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the citation recommendation model over the diagnostic dataset.
        For each CitingSentence, obtain predictions and compare against the ground truth.
        Computes the average Recall@k and MRR@k over all samples.
        
        Returns:
            Dict[str, float]: Dictionary with keys "Recall@10" and "MRR@10" and their averaged values.
        """
        if not self.dataset:
            logger.error("Evaluation dataset is empty. Cannot compute metrics.")
            return {"Recall@10": 0.0, "MRR@10": 0.0}

        recall_list: List[float] = []
        mrr_list: List[float] = []

        total_samples = len(self.dataset)
        logger.info("Starting evaluation on %d samples.", total_samples)

        for idx, citing_sample in enumerate(self.dataset):
            context: str = citing_sample.text
            # Extract ground truth citation from metadata (original_citation stored during citation parsing)
            ground_truth: str = citing_sample.metadata.get("original_citation", "")
            if not ground_truth:
                logger.warning("Sample index %d does not have a ground truth citation in metadata.", idx)
                continue

            # Retrieve candidate citations using the model's forward() method
            try:
                predictions: List[str] = self.model.forward(context)
            except Exception as e:
                logger.error("Error during model prediction for sample index %d: %s", idx, e)
                continue

            if not predictions:
                logger.warning("Model returned empty predictions for sample index %d.", idx)
                recall_value = 0.0
                mrr_value = 0.0
            else:
                recall_value = self.compute_recall(predictions, ground_truth, self.recall_k)
                mrr_value = self.compute_mrr(predictions, ground_truth, self.mrr_k)
            
            recall_list.append(recall_value)
            mrr_list.append(mrr_value)

            if (idx + 1) % 50 == 0:
                logger.info("Processed %d/%d samples.", idx + 1, total_samples)

        # Compute average Recall and MRR over all valid samples (avoid division by zero)
        avg_recall = float(np.mean(recall_list)) if recall_list else 0.0
        avg_mrr = float(np.mean(mrr_list)) if mrr_list else 0.0

        metrics = {"Recall@10": avg_recall, "MRR@10": avg_mrr}
        logger.info("Evaluation complete. Metrics: %s", metrics)
        return metrics


# For module testing: Run evaluation if executed as the main module
if __name__ == "__main__":
    # For demonstration, we create dummy CitingSentence objects and a dummy model.
    try:
        # Attempt to import BM25Model from model.py for dummy evaluation.
        from model import BM25Model
    except ImportError:
        raise ImportError("Failed to import BM25Model from model.py.")

    # Create dummy dataset: list of CitingSentence with dummy ground truth in metadata.
    dummy_dataset: List[CitingSentence] = [
        CitingSentence(
            text="This paper builds on previous work <REF> to improve performance.", 
            token_count=10, 
            citation_position=0.8, 
            preceding_pos="ADJ", 
            following_pos="VERB", 
            metadata={"paper_id": "dummy1", "original_citation": "candidate_2"}
        ),
        CitingSentence(
            text="Recent studies in the field <REF> have shown promising results.", 
            token_count=12, 
            citation_position=0.75, 
            preceding_pos="NOUN", 
            following_pos="AUX", 
            metadata={"paper_id": "dummy2", "original_citation": "candidate_5"}
        ),
    ]

    # Create a dummy corpus for BM25Model evaluation.
    # Here, we create a tokenized dummy corpus aligned with the candidate identifiers used in predictions.
    dummy_corpus = [
        "This is a dummy document for candidate_0".split(),
        "This is a dummy document for candidate_1".split(),
        "This is a dummy document for candidate_2".split(),
        "This is a dummy document for candidate_3".split(),
        "This is a dummy document for candidate_4".split(),
        "This is a dummy document for candidate_5".split(),
        "This is a dummy document for candidate_6".split(),
        "This is a dummy document for candidate_7".split(),
        "This is a dummy document for candidate_8".split(),
        "This is a dummy document for candidate_9".split(),
    ]
    # Initialize the BM25Model with the dummy corpus.
    dummy_model = BM25Model(corpus=dummy_corpus, top_k=10)
    
    # Create Evaluation instance with dummy model and dataset.
    evaluator = Evaluation(model=dummy_model, dataset=dummy_dataset)
    results = evaluator.evaluate()
    
    # Print the evaluation results.
    print("Evaluation Metrics:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
