# Handwritten Digit Classification using Artificial Neural Network (ANN)

This project implements an Artificial Neural Network (ANN) for classifying handwritten digit images from the MNIST dataset. The network is trained to recognize digits (0-9) and predict the label for new, unseen handwritten digit images.

## Project Overview

The goal of this project is to classify images of handwritten digits using a neural network implemented in TensorFlow and Keras. This model is trained to accurately predict digits from the MNIST dataset, achieving high accuracy with a simple feedforward architecture.

## Features

- Loads and preprocesses the MNIST dataset.
- Defines and trains a feedforward ANN with multiple dense layers.
- Evaluates model accuracy on test data.
- Loads pre-trained models for inference on new images.

## Dataset

This project uses the **MNIST** dataset, a large collection of handwritten digits widely used for training and testing in image processing and machine learning. The dataset includes:
- 60,000 training images
- 10,000 testing images

Each image is a 28x28 grayscale image representing a single digit (0–9).

### Loading the Dataset

The dataset files used are in IDX format:
- `train-images.idx3-ubyte` – training images
- `train-labels.idx1-ubyte` – training labels
- `t10k-images.idx3-ubyte` – testing images
- `t10k-labels.idx1-ubyte` – testing labels

## Model Architecture

The ANN model is implemented with TensorFlow and Keras, with the following structure:

1. **Flatten Layer**: Converts 28x28 pixel images to a flat input layer with 784 nodes.
2. **Dense Layer (512 units, ReLU)**: Hidden layer with 512 neurons and ReLU activation.
3. **Dense Layer (256 units, ReLU)**: Hidden layer with 256 neurons and ReLU activation.
4. **Dense Layer (128 units, ReLU)**: Hidden layer with 128 neurons and ReLU activation.
5. **Output Layer (10 units, Softmax)**: Output layer with 10 neurons, each representing a digit class (0-9), using softmax activation for probability distribution.

The model is compiled with:
- **Optimizer**: `adam`
- **Loss**: `sparse_categorical_crossentropy`
- **Metrics**: `accuracy`

### Prerequisites

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- OpenCV (for loading new handwritten images)


