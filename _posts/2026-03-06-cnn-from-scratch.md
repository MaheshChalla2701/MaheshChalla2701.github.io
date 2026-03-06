---
layout: post
title: "Building a CNN from Scratch"
date: 2026-03-06
---

Convolutional Neural Networks (CNNs) are a class of deep neural networks, most commonly applied to analyzing visual imagery. 

They are known for their ability to learn spatial hierarchies of features automatically and adaptively from backpropagation. building one from scratch gives us great insight into how they work.

### What is a Convolution?
A convolution is a mathematical operation on two functions that produces a third function expressing how the shape of one is modified by the other. In the context of CNNs, it's a specialized kind of linear operation.

### Building Blocks of a CNN
To build a CNN from scratch, we need to understand its fundamental building blocks:

1.  **Convolutional Layers:** These are the core building blocks. They apply a set of filters (kernels) to the input image to create feature maps.
2.  **Activation Functions:** Functions like ReLU introduction non-linearity into the network, allowing it to learn complex patterns.
3.  **Pooling Layers:** These layers downsample the feature maps, reducing the spatial dimensions and making the representation more manageable. Max pooling is the most common technique.
4.  **Fully Connected Layers:** At the end of the network, the feature maps are flattened and passed through fully connected layers (like in a standard neural network) to perform the final classification.

### Implementation Steps

Let's break down the implementation process step-by-step:

1.  **Initialize Weights and Biases:** We need to initialize the weights (filters) and biases for the convolutional and fully connected layers.
2.  **Forward Propagation:**
    *   **Convolution Operation:** Implement the math for sliding the filter over the input and calculating the dot product.
    *   **Activation:** Apply the ReLU function to the output of the convolution.
    *   **Pooling Operation:** Implement max pooling to reduce the dimensions.
    *   **Flatten and Dense Layers:** Flatten the output and pass it through the final dense layers for prediction.
3.  **Loss Function:** Define a loss function (e.g., Categorical Cross-Entropy) to measure the error between the prediction and the ground truth.
4.  **Backward Propagation (The hard part):** We need to calculate the gradients of the loss with respect to all the weights and biases in the network. This involves applying the chain rule layer by layer backwards.
    *   **Gradient of fully connected layer.**
    *   **Gradient through pooling layer.**
    *   **Gradient of convolutional layer.**
5.  **Parameter Update:** Update the weights and biases using an optimization algorithm like Stochastic Gradient Descent (SGD) or Adam.

_More complex code and math details will go here as we flesh out the blog._
