# S2ORC Diagnostic Citation Recommendation Benchmark

## Overview

This project orchestrates an end-to-end workflow for a citation recommendation benchmark experiment. It is designed to evaluate the performance of various citation recommendation models (including BM25 as a baseline and several neural models) on a diagnostic dataset. This dataset is constructed from preprocessed sentences derived from the S2ORC (Semantic Scholar Open Research Corpus) and S2AG (Semantic Scholar Academic Graph) datasets. The primary goal is to provide a standardized framework for comparing different approaches to citation recommendation.

## Workflow / Key Features

The experiment follows these key steps:

1.  **Configuration Loading:** Loads all necessary parameters (dataset paths, model configurations, evaluation settings, logging preferences) from `config.yaml` via the `config.py` module.
2.  **Logging Initialization:** Sets up logging to monitor the experiment's progress and capture important events.
3.  **Data Loading and Preprocessing:** Utilizes `DatasetLoader` to load and preprocess data from the S2ORC and S2AG datasets.
4.  **Diagnostic Dataset Construction:** Forms a diagnostic dataset consisting of citing sentences, which serves as the basis for model training and evaluation.
5.  **Model Instantiation:**
    *   Instantiates the selected citation recommendation model based on the configuration.
    *   Supports **BM25** as a baseline, which involves building a candidate corpus from tokenized diagnostic sentences.
    *   Supports neural models such as **NCN (Neural Citation Network)**, **LCR (Local Citation Recommendation)**, and **Galactica**.
6.  **Model Training (for Neural Models):** If a neural model is selected, the `Trainer` module is invoked to handle the training process using the diagnostic dataset and specified hyperparameters.
7.  **Model Evaluation:** The `Evaluation` module assesses the performance of the trained or initialized model on the diagnostic dataset. Key metrics computed include **Recall@10** and **MRR@10**.
8.  **Results Logging:** Logs and prints the final evaluation outcomes, providing insights into the model's effectiveness.

## Project Structure

The project is organized into the following key modules:

*   `main.py`: The main script that orchestrates the entire benchmark experiment from start to finish.
*   `config.py`: A utility module responsible for loading and providing access to the configurations defined in `config.yaml`.
*   `config.yaml`: The central configuration file where users can specify dataset paths, model types, hyperparameters, evaluation parameters, and logging settings.
*   `dataset_loader.py`: Contains the `DatasetLoader` class, which handles the loading, parsing, and preprocessing of the S2ORC and S2AG datasets to create the diagnostic sentences.
*   `model.py`: Defines the base `Model` interface and specific model implementations, including `BM25Model` for the baseline and `NeuralModel` for various neural network-based approaches.
*   `trainer.py`: Includes the `Trainer` class, which manages the training loop, optimization, and checkpointing for neural models.
*   `evaluation.py`: Provides the `Evaluation` class, responsible for calculating and reporting performance metrics like Recall@10 and MRR@10.

## Configuration

All aspects of the experiment are controlled through the `config.yaml` file. This includes:

*   Paths to the S2ORC and S2AG datasets.
*   Selection of the model to be evaluated (e.g., "BM25", "NCN").
*   Hyperparameters for neural models (e.g., learning rate, batch size, embedding dimensions).
*   Evaluation settings (e.g., `recall_k` for Recall@k).
*   Logging level and format.

Please refer to the `config.yaml` file for detailed options and examples.

## Running the Experiment

To run the citation recommendation benchmark experiment, execute the main script from the project's root directory:

```bash
python main.py
```

Ensure that Python is installed and any necessary dependencies are available in your environment. The script will read the configuration from `config.yaml`, run the experiment, and output the evaluation results to the console and log files.

## Evaluation Metrics

The performance of the citation recommendation models is primarily evaluated using the following metrics:

*   **Recall@10 (R@10):** Measures the proportion of actual relevant citations that are found within the top 10 recommendations provided by the model.
*   **Mean Reciprocal Rank@10 (MRR@10):** Calculates the average of the reciprocal ranks of the first correct recommendation for each query, considering only the top 10 recommendations. A higher MRR indicates that the correct citation is, on average, ranked higher by the model.
