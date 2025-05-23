"""
dataset_loader.py

This module implements the DatasetLoader class that is responsible for loading and preprocessing raw
data from the S2ORC full-text dataset and S2AG metadata. It extracts text from documents, segments the text
into sentences using SciSpacy, parses and replaces citation strings with the "<REF>" placeholder,
computes token counts, citation position, and surrounding POS information, and finally creates instances
of the CitingSentence class with associated metadata.

The processing pipeline includes:
  1. Data Loading from disk (S2ORC and S2AG paths provided via config.yaml)
  2. Text extraction and cleaning (trimming, basic normalization)
  3. Sentence segmentation using a SciSpacy model (default: "en_core_sci_sm")
  4. Citation parsing using regex:
       - Valid sentences must have exactly one citation in a standard format.
       - The citation text is replaced with "<REF>", and the original citation is stored separately.
  5. Tokenization, computing token count, and outlier removal via z-score filtering (threshold = 3)
  6. Extraction of normalized citation position and POS tags for tokens surrounding "<REF>"
  7. Creation of CitingSentence objects with metadata and returning a list for subsequent pipeline steps.

The module also logs important events and warnings regarding dropped sentences during processing.
"""

import os
import re
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import spacy

from config import config

# Set up logger using the logging configuration provided in config
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, config["logging"].get("level", "INFO").upper(), logging.INFO))


class CitingSentence:
    """
    CitingSentence stores a preprocessed sentence used for citation recommendation experiments.
    
    Attributes:
        text (str): The processed sentence text with the citation replaced by "<REF>".
        citation_placeholder (str): The placeholder text used for citation, always "<REF>".
        token_count (int): The number of tokens in the processed sentence.
        citation_position (float): The normalized position (0 to 1) of the citation ("<REF>") in the sentence.
        preceding_pos (str): POS tag of the token immediately preceding "<REF>".
        following_pos (str): POS tag of the token immediately following "<REF>".
        metadata (Dict[str, Any]): Additional metadata (e.g., paper id, publication year, field etc.).
    """
    def __init__(self, text: str, token_count: int, citation_position: float,
                 preceding_pos: str, following_pos: str, metadata: Optional[Dict[str, Any]] = None,
                 original_citation: Optional[str] = None):
        self.text: str = text
        self.citation_placeholder: str = "<REF>"
        self.token_count: int = token_count
        self.citation_position: float = citation_position
        self.preceding_pos: str = preceding_pos
        self.following_pos: str = following_pos
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}
        # Store original citation text for future reference if needed.
        self.metadata["original_citation"] = original_citation

    def __repr__(self) -> str:
        return (f"CitingSentence(text='{self.text[:30]}...', token_count={self.token_count}, "
                f"citation_position={self.citation_position:.2f}, preceding_pos='{self.preceding_pos}', "
                f"following_pos='{self.following_pos}', metadata={self.metadata})")


class DatasetLoader:
    """
    DatasetLoader is responsible for loading raw documents and metadata, preprocessing text,
    segmenting sentences, parsing citations, and filtering out outlier sentences.
    
    Public Methods:
        __init__(config: dict): Constructor that loads configuration and initializes resources.
        load_data() -> List[CitingSentence]: Loads data and returns a list of processed CitingSentence objects.
        preprocess_text(text: str) -> str: Performs preliminary cleaning on a text string.
        extract_sentences(document: str) -> List[str]: Performs sentence segmentation on a document.
    """
    # Regex pattern to capture common in-text citations.
    # This pattern matches text enclosed in parentheses that include a 4-digit year
    # and possibly additional punctuation, e.g., "(Smith et al., 2020)".
    CITATION_REGEX: str = r"\([A-Za-z][^()]*\d{4}[a-z]?(?:,\s*[^()]*\d{4}[a-z]?)*\)"

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config_dict
        # Paths for S2ORC full-text dataset and S2AG metadata
        self.s2orc_path: str = self.config["dataset"].get("s2orc_path", "path/to/s2orc/dataset")
        self.s2ag_path: str = self.config["dataset"].get("s2ag_path", "path/to/s2ag/metadata")
        # Sampling threshold (minimum percentage for field filtering, not used in loader but passed downstream)
        self.sampling_threshold: int = self.config["dataset"].get("sampling_threshold", 3)
        # Z-score threshold for outlier removal
        self.zscore_threshold: float = 3.0

        # Initialize SciSpacy model for sentence segmentation and tokenization
        try:
            self.nlp = spacy.load("en_core_sci_sm")
        except Exception as e:
            logger.error("Error loading SciSpacy model 'en_core_sci_sm': %s", e)
            raise e

        # Load S2AG metadata into a pandas DataFrame for later merging with documents
        # We assume the metadata is stored in a CSV file format.
        try:
            self.metadata_df = pd.read_csv(self.s2ag_path)
            # For quick lookup, create a dict mapping from paper id to its metadata
            if "paper_id" in self.metadata_df.columns:
                self.metadata_lookup = self.metadata_df.set_index("paper_id").to_dict("index")
            else:
                self.metadata_lookup = {}
                logger.warning("S2AG metadata does not contain 'paper_id' column; metadata lookup will be empty.")
        except Exception as e:
            logger.error("Error loading S2AG metadata from '%s': %s", self.s2ag_path, e)
            self.metadata_lookup = {}

    def preprocess_text(self, text: str) -> str:
        """
        Performs preliminary cleaning of the raw text.
        
        Args:
            text (str): The raw text extracted from a document.
        
        Returns:
            str: Cleaned text.
        """
        # Basic cleaning: strip extra whitespace and normalize newlines.
        cleaned_text = text.strip().replace('\n', ' ')
        return cleaned_text

    def extract_sentences(self, document: str) -> List[str]:
        """
        Segment the document text into sentences using the SciSpacy model.
        
        Args:
            document (str): The cleaned document text.
        
        Returns:
            List[str]: A list of sentences extracted from the document.
        """
        doc = self.nlp(document)
        sentences: List[str] = []
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            # Ensure sentence ends with proper punctuation (. ? or !)
            if not re.search(r"[.?!]$", sentence_text):
                continue
            # Filter out sentences that likely are footer notes (starting with a digit)
            if re.match(r"^\d", sentence_text):
                continue
            sentences.append(sentence_text)
        return sentences

    def load_data(self) -> List[CitingSentence]:
        """
        Loads and processes raw documents from the S2ORC dataset, performs sentence segmentation,
        citation parsing, and outlier removal. Integrates metadata from the S2AG dataset.

        Returns:
            List[CitingSentence]: A list of processed CitingSentence objects.
        """
        citing_sentences: List[CitingSentence] = []
        token_counts: List[int] = []

        # Process each text file in the S2ORC dataset directory.
        if not os.path.isdir(self.s2orc_path):
            logger.error("S2ORC path '%s' is not a valid directory.", self.s2orc_path)
            return citing_sentences

        file_list = [f for f in os.listdir(self.s2orc_path) if f.endswith(".txt")]
        logger.info("Found %d documents in S2ORC dataset.", len(file_list))
        
        for filename in file_list:
            file_path = os.path.join(self.s2orc_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    raw_text = file.read()
            except Exception as e:
                logger.error("Error reading file '%s': %s", file_path, e)
                continue

            cleaned_text = self.preprocess_text(raw_text)
            # Extract sentences using SciSpacy
            sentences = self.extract_sentences(cleaned_text)
            logger.debug("Document '%s': extracted %d sentences.", filename, len(sentences))

            for sent in sentences:
                # Use regex to find citation candidates
                citation_matches = re.findall(self.CITATION_REGEX, sent)
                if len(citation_matches) != 1:
                    # Filter out sentences with not exactly one citation
                    logger.debug("Sentence dropped due to citation count !=1: '%s'", sent)
                    continue

                original_citation = citation_matches[0]
                # Replace the citation with the placeholder "<REF>"
                sent_replaced = re.sub(self.CITATION_REGEX, "<REF>", sent, count=1)

                # Tokenize the sentence using the spacy model
                doc = self.nlp(sent_replaced)
                tokens = [token for token in doc]
                token_count = len(tokens)
                if token_count == 0:
                    logger.debug("Sentence dropped due to zero tokens: '%s'", sent_replaced)
                    continue

                # Find index of "<REF>" token
                ref_index = None
                for idx, token in enumerate(tokens):
                    if token.text == "<REF>":
                        ref_index = idx
                        break
                if ref_index is None:
                    logger.debug("Sentence dropped because '<REF>' not found after replacement: '%s'", sent_replaced)
                    continue

                citation_position = ref_index / token_count

                # Extract POS for preceding and following tokens, if available.
                preceding_pos = tokens[ref_index - 1].pos_ if ref_index > 0 else ""
                following_pos = tokens[ref_index + 1].pos_ if ref_index < token_count - 1 else ""

                # Merge metadata from S2AG if available based on paper_id (assuming filename without extension is paper_id)
                paper_id = os.path.splitext(filename)[0]
                metadata: Dict[str, Any] = {"paper_id": paper_id}
                if paper_id in self.metadata_lookup:
                    metadata.update(self.metadata_lookup[paper_id])
                else:
                    logger.debug("No S2AG metadata found for paper_id: %s", paper_id)

                # Create CitingSentence object and add to list.
                citing_sentence = CitingSentence(
                    text=sent_replaced,
                    token_count=token_count,
                    citation_position=citation_position,
                    preceding_pos=preceding_pos,
                    following_pos=following_pos,
                    metadata=metadata,
                    original_citation=original_citation
                )
                citing_sentences.append(citing_sentence)
                token_counts.append(token_count)

        logger.info("Total sentences before outlier removal: %d", len(citing_sentences))
        # Outlier Removal based on z-score of token counts
        if token_counts:
            token_array = np.array(token_counts)
            mean_tokens = np.mean(token_array)
            std_tokens = np.std(token_array)
            filtered_sentences: List[CitingSentence] = []
            for cs in citing_sentences:
                z_score = abs(cs.token_count - mean_tokens) / (std_tokens + 1e-6)  # add epsilon to avoid division by zero
                if z_score > self.zscore_threshold:
                    logger.debug("Sentence dropped due to high z-score (%.2f): '%s'", z_score, cs.text)
                    continue
                filtered_sentences.append(cs)
            logger.info("Total sentences after outlier removal: %d", len(filtered_sentences))
            return filtered_sentences
        else:
            logger.warning("No valid sentences were processed from the dataset.")
            return citing_sentences


# For quick module testing
if __name__ == "__main__":
    loader = DatasetLoader(config)
    sentences = loader.load_data()
    logger.info("Loaded %d valid CitingSentence objects.", len(sentences))
    for cs in sentences[:5]:
        logger.info(cs)
