### ⚠️ DISCLAIMER: This repository is not in its finished state. Consider this while using! ⚠️

# GENUINE Project

Welcome to the GENUINE project repository. This document provides an overview of the project, including its purpose, structure, and guidelines for getting started.

## Introduction

The GENUINE project aims to provide a comprehensive suite of models and tools for advanced image analysis and machine learning tasks. It focuses on leveraging deep learning techniques to solve complex problems in various domains.

## Repository Structure

The repository is organized into several key directories:

- **[ModelZoo](https://github.com/SimonBon/GENUINE/tree/main/GENUINE/ModelZoo):** Contains the implementation of various models including GENUINE, Baseline, RetinaNet, GENUINE_B, and GENUINE_E. Each model is designed for specific types of image analysis tasks.
- **[TRAINED_MODELS](https://github.com/SimonBon/GENUINE/tree/main/GENUINE/TRAINED_MODELS):** Includes pre-trained models that are ready to use for evaluation or further training.
- **[data](https://github.com/SimonBon/GENUINE/tree/main/GENUINE/data):** Contains scripts and utilities for data handling, including custom transforms and dataset management.
- **[training](https://github.com/SimonBon/GENUINE/tree/main/GENUINE/training):** Offers training scripts and a training utility module to facilitate the training process of different models.
- **[utils](https://github.com/SimonBon/GENUINE/tree/main/GENUINE/utils):** Provides a collection of utility functions and classes for device management, data visualization, and more.

Additionally, the repository includes Jupyter notebooks such as [embed.ipynb](https://github.com/SimonBon/GENUINE/blob/main/GENUINE/embed.ipynb) and [try_out.ipynb](https://github.com/SimonBon/GENUINE/blob/main/GENUINE/try_out.ipynb) for demonstration and experimentation purposes.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed on your system. The project dependencies are listed in the [requirements.txt](https://github.com/SimonBon/GENUINE/blob/main/requirements.txt) file.

### Installation

1. Clone the repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. To install the GENUINE package locally, run:
   ```bash
   python setup.py install
   ```

### Usage

Refer to the individual model scripts within the [ModelZoo](https://github.com/SimonBon/GENUINE/tree/main/GENUINE/ModelZoo) directory for details on using each model. Training scripts located in the [training_files](https://github.com/SimonBon/GENUINE/tree/main/GENUINE/training_files) directory provide examples of how to train the models with your data.

## Contributing

Contributions to the GENUINE project are welcome. Please refer to the contributing guidelines for more information on how to submit pull requests, report issues, or request features.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/SimonBon/GENUINE/blob/main/LICENSE) file for details.
