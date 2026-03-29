# Neural Networks

Neural networks are computing systems inspired by biological neural networks in the brain. They consist of layers of interconnected nodes (neurons) that process information using connectionist approaches.

## Architecture

A basic neural network has three types of layers:

- **Input layer**: Receives the raw data features
- **Hidden layers**: Perform transformations through weighted connections and activation functions
- **Output layer**: Produces the final prediction or classification

Each connection has a weight, and each neuron applies an activation function (ReLU, sigmoid, tanh) to its weighted sum of inputs. The choice of activation function significantly impacts training dynamics.

## Training

Neural networks learn through backpropagation, which computes gradients of the loss function with respect to each weight using the chain rule of calculus. These gradients guide the weight updates via gradient descent.

Key training considerations include:
- **Learning rate**: Too high causes divergence, too low causes slow convergence
- **Batch size**: Affects gradient estimation quality and training speed
- **Weight initialization**: Xavier/Glorot or He initialization helps prevent vanishing/exploding gradients

## Types of Neural Networks

- **Feedforward networks**: Information flows in one direction
- **Convolutional neural networks (CNNs)**: Excel at spatial data like images
- **Recurrent neural networks (RNNs)**: Handle sequential data like text and time series
- **Graph neural networks (GNNs)**: Operate on graph-structured data

Neural networks are the building blocks of [[deep_learning]] and are rooted in [[machine_learning]] theory. Modern architectures like [[transformers]] have revolutionized the field.

#neural-networks #deep-learning #ai
