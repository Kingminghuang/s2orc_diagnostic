"""
model.py

This module defines the abstract base Model class and its concrete implementations for citation recommendation.
It includes:
  - Model (abstract base class) with forward() and retrieve_candidates() methods.
  - BM25Model: A retrieval-based model using BM25 from the rank_bm25 library.
  - NeuralModel: A stub implementation of a neural citation recommendation model using HuggingFace transformers and torch.

Configuration is loaded from config.py; default values are provided for settings.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

import numpy as np

# Import BM25 model from rank_bm25
from rank_bm25 import BM25Okapi

# Import torch and transformers for NeuralModel
import torch
from transformers import AutoTokenizer, AutoModel

# Import configuration from config module
from config import config


class Model(ABC):
    """
    Abstract base class that defines the interface for citation recommendation models.
    Concrete implementation must implement the following methods:

    Methods:
        forward(context: str) -> List[str]:
            Process input context and return a ranked list of candidate citation identifiers.
        retrieve_candidates(context: str) -> List[str]:
            Retrieve candidate citations for a given context.
    """

    @abstractmethod
    def forward(self, context: str) -> List[str]:
        """
        Process the input context and produces a ranked list of candidate citation identifiers.

        Args:
            context (str): A citing sentence with "<REF>" as a placeholder for the citation.

        Returns:
            List[str]: Ranked list of candidate citation identifiers.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_candidates(self, context: str) -> List[str]:
        """
        Retrieve candidate citations for the provided input context.

        Args:
            context (str): A citing sentence with "<REF>" in place of the citation.

        Returns:
            List[str]: Candidate citation identifiers.
        """
        raise NotImplementedError


class BM25Model(Model):
    """
    BM25Model implements the Model interface using the BM25 ranking function from the rank_bm25 library.

    Attributes:
        corpus (List[List[str]]): List of tokenized candidate documents.
        bm25 (BM25Okapi): BM25 model computed on the candidate corpus.
        top_k (int): The maximum number of candidates to return.
    """

    def __init__(self, corpus: List[List[str]], top_k: int = config["evaluation"].get("recall_k", 10)) -> None:
        """
        Initializes the BM25Model with the provided preprocessed corpus.

        Args:
            corpus (List[List[str]]): Pre-tokenized candidate documents.
            top_k (int): Number of top candidates for retrieval.
        """
        if not corpus or not isinstance(corpus, list):
            raise ValueError("Corpus must be a non-empty list of tokenized documents.")
        self.corpus: List[List[str]] = corpus
        self.top_k: int = top_k
        self.bm25: BM25Okapi = BM25Okapi(corpus)
    
    def forward(self, context: str) -> List[str]:
        """
        BM25 forward pass simply retrieves candidates based on the context.

        Args:
            context (str): The citing context with "<REF>" placeholder.

        Returns:
            List[str]: A ranked list of candidate citation identifiers.
        """
        return self.retrieve_candidates(context)
    
    def retrieve_candidates(self, context: str) -> List[str]:
        """
        Tokenizes the input context using a simple whitespace split (should match corpus tokenization)
        and retrieves top candidates using BM25's get_top_n.

        Args:
            context (str): The citing sentence with "<REF>" placeholder.

        Returns:
            List[str]: Top candidate citations.
        """
        if not context or not isinstance(context, str):
            raise ValueError("Input context must be a non-empty string.")
        
        # Simple whitespace tokenization; ensure consistency with corpus tokenization
        query_tokens: List[str] = context.split()
        # BM25 get_top_n expects a list of documents; here we simulate candidate IDs as indices cast to str.
        top_candidates: List[Any] = self.bm25.get_top_n(query_tokens, self.corpus, n=self.top_k)
        # For demonstration, we return string representations of candidate indices.
        # In a real implementation, each document should have a corresponding unique citation identifier.
        candidate_ids: List[str] = [str(idx) for idx, doc in enumerate(self.corpus) if doc in top_candidates][:self.top_k]
        return candidate_ids


class NeuralModel(Model):
    """
    NeuralModel provides a stub implementation of a neural citation recommendation model.
    It uses HuggingFace Transformers and torch for inference, but detailed model architecture and
    hyperparameters are kept flexible due to unspecified details in the paper.

    Attributes:
        tokenizer: Pretrained tokenizer.
        model: Pretrained transformer model.
        device: Torch device (CPU/GPU) used for computation.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initializes the NeuralModel using parameters from configuration.
        
        Args:
            params (Dict[str, Any]): Hyperparameters and model settings from configuration.
        """
        # Choose a default model name if not provided via parameters.
        default_model_name: str = "bert-base-uncased"
        model_name: str = params.get("model_name", default_model_name)
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load transformer model '{model_name}': {e}")
    
    def forward(self, context: str) -> List[str]:
        """
        Processes the citing context through the neural transformer model.
        This stub implementation tokenizes the input and performs a forward pass,
        then returns a dummy list of candidate citation identifiers.

        Args:
            context (str): The input citing context with "<REF>" placeholder.

        Returns:
            List[str]: Ranked candidate citation identifiers.
        """
        if not context or not isinstance(context, str):
            raise ValueError("Input context must be a non-empty string.")
        
        # Tokenize the input context; add batch dimension.
        inputs = self.tokenizer(context, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # In real implementation, one would use outputs to compute similarity scores.
        # Here we simulate by returning a dummy candidate list.
        dummy_candidates: List[str] = ["candidate_1", "candidate_2", "candidate_3", "candidate_4", "candidate_5",
                                       "candidate_6", "candidate_7", "candidate_8", "candidate_9", "candidate_10"]

        return dummy_candidates

    def retrieve_candidates(self, context: str) -> List[str]:
        """
        Retrieves candidate citations using the neural model. In this stub, it simply calls forward.

        Args:
            context (str): The citing sentence with "<REF>" placeholder.

        Returns:
            List[str]: Ranked candidate citations.
        """
        return self.forward(context)
