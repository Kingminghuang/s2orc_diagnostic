"""
trainer.py

This module implements the Trainer class responsible for training/fine-tuning neural citation recommendation
models (e.g., NCN, LCR, Galactica) as specified by the benchmark paper. If the selected model is BM25 (a retrieval-based baseline),
the Trainer will skip training. The Trainer class supports training over a dataset of CitingSentence objects using
a simple training loop, computes a dummy ranking loss (using MSELoss as a placeholder), performs backpropagation,
and provides a validate() method to compute validation loss. Training hyperparameters (learning_rate, batch_size, epochs, optimizer)
are loaded from the configuration (config.yaml) via config.py. Sensible defaults are used when these values are not provided.

Usage:
    from trainer import Trainer
    trainer = Trainer(model, training_dataset, validation_dataset)
    trainer.train()
    metrics = trainer.validate()
"""

import os
import logging
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from config import config
from model import Model

# Set up logger based on configuration
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, config["logging"].get("level", "INFO").upper(), logging.INFO))


class CitingSentenceDataset(Dataset):
    """
    A simple PyTorch Dataset to wrap a list of CitingSentence objects.
    It returns the preprocessed citing sentence text (with "<REF>" in place of the citation)
    and its associated metadata if needed.
    """
    def __init__(self, citing_sentences: List[Any]) -> None:
        """
        Args:
            citing_sentences (List[Any]): List of CitingSentence objects.
        """
        self.citing_sentences = citing_sentences

    def __len__(self) -> int:
        return len(self.citing_sentences)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.citing_sentences[index]
        # We will use only the text for training and ignore other metadata here
        return {"context": item.text, "metadata": item.metadata}


class Trainer:
    """
    Trainer is responsible for managing the training (and optional fine-tuning) of neural citation recommendation models.
    It supports a training loop, loss computation, backpropagation, optimizer updates, and checkpointing.
    
    Attributes:
        model (Model): An instance of a Model subclass (typically NeuralModel).
        training_dataset (List): List of CitingSentence objects for training.
        validation_dataset (Optional[List]): List of CitingSentence objects for validation (can be None).
        learning_rate (float): Learning rate for optimization.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        device (torch.device): Device (CPU/GPU) for training.
    """
    def __init__(self, model: Model, training_dataset: List[Any], validation_dataset: Optional[List[Any]] = None) -> None:
        """
        Initialize the Trainer with the model and training (optionally, validation) dataset.
        
        Args:
            model (Model): Instance of the citation recommendation model.
            training_dataset (List): List of CitingSentence objects (preprocessed) for training.
            validation_dataset (Optional[List]): List of CitingSentence objects for validation.
        """
        self.model = model
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset

        # Retrieve training hyperparameters from configuration or use defaults.
        training_config: Dict[str, Any] = config.get("training", {})
        self.learning_rate: float = training_config.get("learning_rate") if training_config.get("learning_rate") is not None else 0.001
        self.batch_size: int = training_config.get("batch_size") if training_config.get("batch_size") is not None else 32
        self.epochs: int = training_config.get("epochs") if training_config.get("epochs") is not None else 10

        # Determine optimizer type; if not specified, use Adam.
        optimizer_choice = training_config.get("optimizer") if training_config.get("optimizer") is not None else "Adam"

        # Check if the model type is BM25 (non-trainable baseline)
        if config["model"]["type"].upper() == "BM25":
            logger.info("BM25 model selected as baseline. Training is not applicable for BM25. Skipping optimizer initialization.")
            self.optimizer = None
            self.loss_fn = None
        else:
            # Initialize optimizer for neural models
            if optimizer_choice.upper() == "ADAM":
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            else:
                # Fallback to Adam optimizer if unknown optimizer is provided.
                logger.warning("Unknown optimizer specified; falling back to Adam.")
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            # Use a placeholder loss function. 
            # In practice, a suitable ranking loss should replace this (e.g., margin ranking loss).
            self.loss_fn = nn.MSELoss()

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # For neural models, ensure the model is moved to the correct device.
        if hasattr(self.model, "to"):
            self.model.to(self.device)

        # Create DataLoader for training dataset
        self.train_loader = DataLoader(
            dataset=CitingSentenceDataset(self.training_dataset),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        # Create DataLoader for validation dataset if provided
        if self.validation_dataset is not None:
            self.val_loader = DataLoader(
                dataset=CitingSentenceDataset(self.validation_dataset),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn
            )
        else:
            self.val_loader = None

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """
        Custom collate function to bundle the batch data.

        Args:
            batch (List[Dict[str, Any]]): List of items from the dataset.

        Returns:
            Dict[str, List[Any]]: Dictionary with keys "context" and "metadata" containing respective lists.
        """
        contexts = [item["context"] for item in batch]
        metadata = [item["metadata"] for item in batch]
        return {"context": contexts, "metadata": metadata}

    def train(self) -> None:
        """
        Train the neural model over the training dataset for a specified number of epochs.
        For BM25, training is skipped.
        """
        # If the selected model is BM25, skip training.
        if config["model"]["type"].upper() == "BM25":
            logger.info("BM25 model selected as baseline. Skipping training loop for BM25.")
            return

        logger.info("Starting training for %d epochs with batch size %d and learning rate %f.",
                    self.epochs, self.batch_size, self.learning_rate)

        # Set the model to training mode.
        self.model.train()

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            for batch in self.train_loader:
                contexts: List[str] = batch["context"]
                # For batching, we process each context individually within the batch.
                batch_loss = 0.0
                self.optimizer.zero_grad()

                # Process each context in the batch.
                for context in contexts:
                    # Forward pass - for neural models, assume forward returns a tensor representation.
                    # Here, we call model.forward() and then simulate a dummy target.
                    # In practice, the forward() should provide logits from which a retrieval loss is computed.
                    prediction = self.model.forward(context)
                    # Simulate a dummy target tensor (zeros) of same shape as prediction.
                    # Since our default NeuralModel stub returns a list of candidate ids, we assume real models return tensors.
                    # For dummy training, we create a tensor with a fixed value.
                    # For demonstration, we use a scalar tensor.
                    pred_tensor = torch.tensor(0.5, device=self.device)
                    target_tensor = torch.tensor(0.0, device=self.device)
                    loss = self.loss_fn(pred_tensor, target_tensor)
                    loss.backward()
                    batch_loss += loss.item()

                # After processing batch contexts, update weights.
                self.optimizer.step()
                epoch_loss += batch_loss
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info("Epoch [%d/%d] - Average Training Loss: %.4f", epoch, self.epochs, avg_loss)

            # Optionally, perform validation at the end of each epoch.
            if self.val_loader is not None:
                val_metrics = self.validate()
                logger.info("Epoch [%d/%d] - Validation Metrics: %s", epoch, self.epochs, val_metrics)

            # Save a checkpoint after each epoch.
            checkpoint_path = os.path.join("checkpoints", f"model_epoch_{epoch}.pt")
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(self.model.state_dict(), checkpoint_path)
            logger.info("Saved model checkpoint at: %s", checkpoint_path)

    def validate(self) -> Dict[str, float]:
        """
        Validate the neural model on the validation dataset.
        Returns a dictionary of key metrics (e.g., average validation loss).

        Returns:
            Dict[str, float]: Dictionary containing validation metrics.
        """
        if self.val_loader is None:
            logger.warning("No validation dataset provided. Skipping validation.")
            return {}

        self.model.eval()
        total_loss = 0.0
        num_samples = 0

        # Disable gradient calculations for validation.
        with torch.no_grad():
            for batch in self.val_loader:
                contexts: List[str] = batch["context"]
                for context in contexts:
                    prediction = self.model.forward(context)
                    pred_tensor = torch.tensor(0.5, device=self.device)
                    target_tensor = torch.tensor(0.0, device=self.device)
                    loss = self.loss_fn(pred_tensor, target_tensor)
                    total_loss += loss.item()
                    num_samples += 1

        avg_val_loss = total_loss / max(num_samples, 1)
        metrics = {"validation_loss": avg_val_loss}

        # Set the model back to train mode after validation.
        self.model.train()
        return metrics


# For testing the Trainer module independently
if __name__ == "__main__":
    # Note: This testing code assumes that a list of CitingSentence objects is available.
    # In a full system, the DatasetLoader would provide this list.
    from dataset_loader import CitingSentence

    # Create dummy CitingSentence objects for demonstration.
    dummy_sentences = [
        CitingSentence(text="This study shows that <REF> has significant impact.", token_count=8, citation_position=0.75, preceding_pos="NOUN", following_pos="VERB", metadata={"paper_id": "dummy1"}),
        CitingSentence(text="Recent advances in <REF> provide new insights.", token_count=7, citation_position=0.57, preceding_pos="ADJ", following_pos="NOUN", metadata={"paper_id": "dummy2"})
    ]

    # Import a dummy/model instance. For BM25, training should be skipped.
    # Here we check configuration to decide whether to simulate a neural training session.
    model_type = config["model"]["type"].upper()
    if model_type == "BM25":
        from model import BM25Model
        # For BM25Model, create a dummy corpus (list of tokenized documents).
        dummy_corpus = [["This", "is", "a", "dummy", "document"],
                        ["Another", "dummy", "document", "for", "BM25"]]
        model_instance = BM25Model(corpus=dummy_corpus)
    else:
        from model import NeuralModel
        # Use default hyperparameters for NeuralModel.
        model_instance = NeuralModel(params=config["model"].get("hyperparameters", {}))

    # Initialize Trainer with dummy training dataset and no validation dataset.
    trainer = Trainer(model=model_instance, training_dataset=dummy_sentences, validation_dataset=dummy_sentences)
    trainer.train()
    val_results = trainer.validate()
    logger.info("Final validation metrics: %s", val_results)
