Bot-IoT Evaluation Pipeline

A Reproducible Machine Learning Workflow for Intrusion Detection Baselines

Overview

This repository implements a complete baseline evaluation workflow for the Bot-IoT dataset, focusing on data preprocessing, model inference, and performance reporting. The objective is to provide a transparent and replicable baseline that can be used for comparison against advanced models, including deep learning and hybrid architectures.

The pipeline loads a preprocessed evaluation CSV, applies the trained model to generate predictions, and produces formal evaluation artefacts including classification reports, confusion matrices, plots, and tabular summaries.

Key Features

Clean, reproducible ML evaluation pipeline.

Support for large-scale Bot-IoT CSV files.

Automated classification metrics and reporting.

Confusion matrix visualization using matplotlib.

Strict reproducibility with deterministic configuration.

Modular code structure suited for research, enterprise, and academic benchmarking.

Project Structure
project_root/
│
├── data/
│   └── bot_iot/
│       └── bot_iot_eval.csv          # Preprocessed evaluation dataset
│
├── models/
│   └── model.pkl                     # Trained model (placeholder)
│
├── results/
│   ├── reports/
│   │   └── classification_report.txt
│   └── figures/
│       └── confusion_matrix.png
│
├── src/
│   ├── load_data.py
│   ├── evaluate.py
│   └── utils.py
│
└── README.md

Requirements

Install project dependencies using:

pip install -r requirements.txt


Typical dependencies include:

Python 3.8+

pandas

numpy

scikit-learn

matplotlib

joblib or pickle (for model loading)

Dataset

This workflow assumes the availability of the preprocessed evaluation CSV:

C:\Desktop\Bot-Lot dataset\notebook\bot_iot_eval.csv


The dataset must include numerical or fully encoded features, as well as a ground-truth label column.

Running the Evaluation Pipeline
1. Load the Evaluation Dataset

The evaluation script automatically loads the specified CSV file:

python src/load_data.py

2. Run the Model Evaluation

Execute:

python src/evaluate.py


The script will:

Load the model checkpoint.

Generate predictions on the evaluation dataset.

Produce a classification report.

Save a confusion matrix.

Persist all outputs to the results/ directory.

Outputs
1. Classification Report

A metrics summary including Precision, Recall, F1 Score, and Support for each class.
Stored under:

results/reports/classification_report.txt

2. Confusion Matrix

A visual confusion matrix (matplotlib) showing class distribution and correctness.
Stored under:

results/figures/confusion_matrix.png


This plot is intentionally placed after the classification performance to provide a visual confirmation that all attack instances were correctly classified with no misclassifications if that is the case.

3. Console Output

All metrics are printed directly to the terminal as well for rapid debugging.

Model Notes

The repository is model-agnostic. Any scikit-learn compatible estimator can be loaded using joblib or pickle. To replace the trained model, simply update:

models/model.pkl


No changes to the evaluation pipeline are required unless the input schema changes.

Reproducibility

To ensure consistent results across systems, the pipeline includes:

Fixed random seeds where applicable.

Deterministic scikit-learn configurations.

Clearly defined preprocessing and evaluation stages.

Version-pinned dependencies through requirements.txt.

Extending the Project
Add Additional Models

Place new models in models/ and update the path in the evaluation script.

Add Advanced Plots

Extend evaluate.py to include:

ROC-AUC curves

Precision-Recall curves

Per-class bar charts

Feature importance analysis

Integrate Deep Learning

The modular architecture allows seamless extension to TensorFlow or PyTorch models.

Contact and Contribution

Pull requests, issue reports, and enhancements are welcome.
Please follow the existing structure and maintain comprehensive documentation for all modifications.
