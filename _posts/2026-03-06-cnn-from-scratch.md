---
layout: post
title: "Building a CNN from Scratch"
date: 2026-03-06
description: "A deep dive into the architecture and mathematics of Convolutional Neural Networks, built from the ground up."
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

### 0. Data Loading and Preprocessing

Before training a CNN, the underlying dataset needs to be heavily prepared. For real-world image classification tasks (like pneumonia detection from chest X-rays), this typically involves several meticulous steps:

Using tools like OpenCV (`cv2`), raw images are read from categorized folders, converted to strictly grayscale to reduce computational overhead, and then resized to fixed, standard dimensions (e.g., 150x150) so they can be stacked into matrices.

```python
import cv2
import os
import numpy as np

training_data = []
categories = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = 150
datadir = "path/to/dataset/train"

# 1. Reading and Standardizing Images
for category in categories:
    path = os.path.join(datadir, category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            if img_array is None:
                continue
            img_resize = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([img_resize, class_num])
        except Exception as e:
            pass
```

After loading the images, it's crucial to randomize the dataset so the neural network doesn't blindly memorize the sequential order of the categories. For convolutions, the linear list of pooled images must be reshaped into a definite `(batch_size, width, height, channels)` format. Because our optimized images are grayscale, the `channels` value is `1`.

Finally, caching datasets containing the image features (`X`) and numerical labels (`Y`) into serializers (like `.pickle`) saves massive amounts of time on subsequent training runs.

```python
import random
import pickle

# 2. Shuffling
random.shuffle(training_data)

x, y = [], []
for feature, label in training_data:
    x.append(feature)
    y.append(label)

# 3. Reshaping for Convolutions
X = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = np.array(y)

# 4. Caching for Efficiency
with open("X.pickle", "wb") as pickle_out:
    pickle.dump(X, pickle_out)
with open("y.pickle", "wb") as pickle_out:
    pickle.dump(Y, pickle_out)
```

### 1. Neural Network Architecture

After loading the images, we build the actual neural network. In a from-scratch implementation without frameworks like TensorFlow, we define the mathematical operations explicitly.

The data (`X`) originally holds 2D structures (the images). For a standard Multi-Layer Perceptron (MLP) classifier, we first flatten these arrays into 1D vectors (`X_flat`). We then define a network architecture with initialized weights and biases:
1.  **Input Layer**: Sized according to the flattened image dimensions.
2.  **Hidden Layer (Dense)**: A fully-connected layer (e.g., 64 neurons) using a **ReLU** (Rectified Linear Unit) activation function to introduce non-linearity.
3.  **Output Layer (Dense)**: A final fully-connected layer leading to a single neuron (for binary classification, like Normal vs. Pneumonia) with a **Sigmoid** activation function to squish outputs natively between 0 and 1.

```python
# Flatten the input images for the MLP
X_flat = X.reshape(X.shape[0], -1)

input_size = X_flat.shape[1] 
hidden_size = 64
output_size = 1

# Initialize weights and biases
weights_input_hidden = np.random.randn(input_size, hidden_size)
weights_hidden_output = np.random.randn(hidden_size, output_size)
bias_hidden = np.zeros((1, hidden_size))
bias_output = np.zeros((1, output_size))
```

### 2. The Training Loop (Forward and Backward Propagation)

The magic of deep learning happens during the training loop. Over several epochs, the network iteratively refines its internal weights.

**Forward Propagation**
For every image, we push the data forward through the network:
*   Calculate the dot product of inputs and hidden weights, add the hidden bias, and pass through ReLU.
*   Calculate the dot product of hidden outputs and output weights, add the output bias, and pass through Sigmoid to get our `predicted_output`.

Mathematically, this looks like:
*   $Z_{hidden} = X \cdot W_{input\_hidden} + B_{hidden}$
*   $A_{hidden} = ReLU(Z_{hidden}) = \max(0, Z_{hidden})$
*   $Z_{output} = A_{hidden} \cdot W_{hidden\_output} + B_{output}$
*   $\hat{y} = Sigmoid(Z_{output}) = \frac{1}{1 + e^{-Z_{output}}}$

**Calculating Loss**
We measure how wrong the prediction was using **Binary Cross-Entropy Loss**. This penalizes the model heavily if it confidently predicts the wrong class.

$$ Loss = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] $$

**Backward Propagation (The Math)**
This is where calculus kicks in (the Chain Rule). We compute the error gradients backwards through the network to figure out how much each weight contributed to the loss:
1.  Find the derivative of the error with respect to the output layer's Sigmoid function: $dZ_{output} = \hat{y} - y$
2.  Propagate that error back to the hidden layer by multiplying it by the hidden layer's weights: $dA_{hidden} = dZ_{output} \cdot W_{hidden\_output}^T$
3.  Find the derivative with respect to the hidden layer's ReLU function: $dZ_{hidden} = dA_{hidden} * ReLU'(Z_{hidden})$

From these, we calculate the gradients for our weights and biases:
*   $dW_{hidden\_output} = A_{hidden}^T \cdot dZ_{output}$
*   $dB_{output} = \sum dZ_{output}$
*   $dW_{input\_hidden} = X^T \cdot dZ_{hidden}$
*   $dB_{hidden} = \sum dZ_{hidden}$

**Updating Parameters**
Finally, we update all weights and biases by subtracting a tiny fraction of their respective gradients (determined by the `learning_rate` or $\alpha$). This nudges the parameters in the exact direction needed to minimize the loss on the next pass.

*   $W_{new} = W_{old} - \alpha \cdot dW$
*   $B_{new} = B_{old} - \alpha \cdot dB$

```python
learning_rate = 0.01
epochs = 10

for epoch in range(epochs):
    for i in range(X.shape[0]):
        # --- Forward Pass ---
        hidden_layer_activation = np.dot(X_flat[i], weights_input_hidden) + bias_hidden
        hidden_layer_output = relu(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        predicted_output = sigmoid(output_layer_activation)

        # --- Calculate Error ---
        loss = -(y[i] * np.log(predicted_output + 1e-8) + (1 - y[i]) * np.log(1 - predicted_output + 1e-8))
        error = predicted_output - y[i]
        
        # --- Backward Propagation ---
        d_predicted_output = error * sigmoid_derivative(predicted_output)
        error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * relu_derivative(hidden_layer_output)
        
        # --- Update Weights and Biases ---
        weights_hidden_output -= hidden_layer_output.reshape(hidden_size, 1).dot(d_predicted_output.reshape(1, output_size)) * learning_rate
        weights_input_hidden -= X_flat[i].reshape(input_size, 1).dot(d_hidden_layer.reshape(1, hidden_size)) * learning_rate

        bias_output -= learning_rate * d_predicted_output
        bias_hidden -= learning_rate * d_hidden_layer
```

After iterating through the epochs, we save the trained weights and biases arrays (again, using `pickle`) so the finalized model can be loaded later to make predictions on entirely new X-rays!

### 3. Making Predictions (Inference)

Once our model is trained and the ideal parameters are saved, predicting the diagnosis for a new, unseen chest X-ray is computationally cheap.

We load the target image from the disk, preprocess it in the *exact same way* as we did our training data (grayscale conversion and resizing back to 150x150), and map it through our learned weights and biases. Keep in mind that for this prediction step, we strictly run a forward pass—no backpropagation or loss calculations are necessary!

The sequence of mathematical operations for a single new flattened image $X_{new}$ is:
1.  **Hidden Layer Combination & Activation:** $Z_{hidden} = X_{new} \cdot W_{input\_hidden} + B_{hidden} \xrightarrow{\text{ReLU}} A_{hidden} = \max(0, Z_{hidden})$
2.  **Output Layer Combination & Activation:** $Z_{output} = A_{hidden} \cdot W_{hidden\_output} + B_{output} \xrightarrow{\text{Sigmoid}} \hat{y} = \frac{1}{1 + e^{-Z_{output}}}$

Here is the code executing that exact inference logic:

```python
import cv2
import numpy as np
import pickle

# Load the trained model weights
with open("weights_input_hidden.pickle", "rb") as f:
    weights_input_hidden = pickle.load(f)
with open("weights_hidden_output.pickle", "rb") as f:
    weights_hidden_output = pickle.load(f)
with open("bias_output.pickle", "rb") as f:
    bias_output = pickle.load(f)
with open("bias_hidden.pickle", "rb") as f:
    bias_hidden = pickle.load(f)

# Load and preprocess the new image
img_array = cv2.imread("new_xray.jpg", cv2.IMREAD_GRAYSCALE)
img_resize = cv2.resize(img_array, (150, 150))
img_normalized = img_resize / 255.0

# Flatten identically to training
img_flat = img_normalized.reshape(1, -1) 

# Forward Pass (Inference)
hidden_layer_activation = np.dot(img_flat, weights_input_hidden) + bias_hidden
hidden_layer_output = np.maximum(0, hidden_layer_activation) # ReLU

output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output) + bias_output

# Sigmoid bounded output
predicted_output = 1 / (1 + np.exp(-np.clip(output_layer_activation, -500, 500)))
pred_val = predicted_output[0][0]

print(f"Normal Probability: {1.0 - pred_val:.4f}")
print(f"Pneumonia Probability: {pred_val:.4f}")
```

This single forward pass matrix multiplication provides lightning-fast inferences without the heavy calculation burdens of backpropagation.
