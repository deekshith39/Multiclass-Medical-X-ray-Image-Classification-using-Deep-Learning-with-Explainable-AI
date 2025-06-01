
# Multiclass Medical X-ray Image Classification using Deep Learning with Explainable AI

This repository presents a major project focused on the classification of medical X-ray images into multiple categories using deep learning techniques, complemented by Explainable AI (XAI) methods to enhance model interpretability.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Explainable AI Techniques](#explainable-ai-techniques)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The primary objective of this project is to develop a deep learning-based system capable of classifying medical X-ray images into multiple disease categories. Given the critical nature of medical diagnoses, the project also integrates Explainable AI techniques to provide insights into the model's decision-making process, thereby enhancing trust and reliability.

## Dataset

The project utilizes a curated dataset of medical X-ray images, encompassing various classes corresponding to different medical conditions. The dataset has been preprocessed and organized for training and evaluation purposes.

## Model Architectures

Several deep learning architectures have been explored and implemented in this project:

- **ResNet**: A deep residual network that addresses the vanishing gradient problem, allowing for the training of very deep networks.
- **VGG**: A convolutional neural network known for its simplicity and depth, using small convolutional filters.
- **Xception**: An architecture that relies on depthwise separable convolutions, offering a balance between performance and computational efficiency.

Each model has been trained and evaluated to determine its effectiveness in classifying medical X-ray images.

## Explainable AI Techniques

To interpret and visualize the decision-making process of the deep learning models, Explainable AI techniques have been employed. These methods help in highlighting the regions of the X-ray images that the models focus on while making predictions, thereby providing transparency and aiding in clinical validation.

## Results

The performance of each model has been evaluated using standard metrics such as accuracy, precision, recall, and F1-score. The `comparison.ipynb` notebook provides a comprehensive comparison of the models' performances.

Additionally, the integration of Explainable AI techniques offers visual interpretations of the models' predictions, aiding in understanding and validating the results.

## Project Structure

The repository is organized as follows:

- `Resnet.ipynb`: Implementation and training of the ResNet model.
- `VGG.ipynb`: Implementation and training of the VGG model.
- `xception.ipynb`: Implementation and training of the Xception model.
- `comparison.ipynb`: Comparative analysis of all implemented models.
- `preprocess.ipynb`: Data preprocessing steps.
- `Research Paper IJRASET.pdf`: Detailed documentation of the project, methodologies, and findings.
- `config.yml`: Configuration file containing parameters and settings.
- `processed/`: Directory containing processed data.
- `graphs/`: Directory containing generated graphs and visualizations.
- `ui/`: Directory related to the user interface components (if any).

## Contributing

Contributions are welcome! If you have suggestions, improvements, or encounter issues, please open an issue or submit a pull request.
