# Info441
Info441: Machine Learning Projects
---

## Exercise 1: Perceptron Learning Algorithm for OR Problem

The perceptron is a fundamental building block of neural networks. It's a linear classifier that learns a decision boundary.

### Program Implementation

You'll need to implement the perceptron learning rule. This involves:

1.  **Initialization**: Randomly initialize weights ($w$) and bias ($b$).
2.  **Activation Function**: Use a step function (e.g., $f(x) = 1$ if $x \ge 0$, else $0$).
3.  **Learning Rule**: For each training example, update weights and bias based on the error:
    * $w_{new} = w_{old} + \alpha \times (target - output) \times input$
    * $b_{new} = b_{old} + \alpha \times (target - output)$
    Where $\alpha$ is the learning rate.
4.  **Epochs**: Repeat the learning process for a fixed number of epochs or until convergence (error is zero for all training examples).

### Solving the OR Problem

The **OR problem** is linearly separable, making it suitable for a perceptron.

| x | y | x OR y |
|---|---|--------|
| 0 | 0 | 0      |
| 0 | 1 | 1      |
| 1 | 0 | 1      |
| 1 | 1 | 1      |

Your program will iterate through these examples, adjust weights, and eventually learn the correct classification.

### Learning Error

For each cycle (epoch), calculate the **sum of squared errors** or the **number of misclassified examples** to track the learning progress. This error should decrease over time as the perceptron learns.

---

## Exercise 2: Gradient Backpropagation for XOR Problem

The **XOR problem** is not linearly separable, meaning a simple perceptron cannot solve it. This is where multi-layered neural networks with backpropagation come in.

### Program Implementation

You'll need to build a neural network with at least one hidden layer and implement the backpropagation algorithm. Key components:

1.  **Network Structure**:
    * Input layer (2 neurons for x, y)
    * At least one hidden layer (e.g., 2 or 3 neurons)
    * Output layer (1 neuron for x XOR y)
2.  **Activation Function**: Use a non-linear activation function like the **sigmoid function** ($f(x) = 1 / (1 + e^{-x})$) for hidden and output layers.
3.  **Forward Pass**: Calculate outputs for each layer, from input to output.
4.  **Backward Pass (Backpropagation)**:
    * Calculate the error at the output layer.
    * Propagate the error backward through the network, adjusting weights and biases using the **gradient descent** rule. This involves calculating the derivative of the activation function.
    * Weight update: $\Delta w_{ij} = -\alpha \frac{\partial E}{\partial w_{ij}}$, where $E$ is the error.

### Solving the XOR Problem

| x | y | x XOR y |
|---|---|---------|
| 0 | 0 | 0       |
| 0 | 1 | 1       |
| 1 | 0 | 1       |
| 1 | 1 | 0       |

Train your network using these four examples. The hidden layer allows the network to learn non-linear decision boundaries.

### Learning Error and Testing

As in Exercise 1, track the **mean squared error (MSE)** for each epoch. After training, **test** your network with the XOR inputs to ensure it predicts the correct outputs.

---

## Exercise 3: Gradient Backpropagation for Digit Recognition (0-9)

This exercise expands on backpropagation for a multi-class classification problem.

### Network Input and Output

* **Input**: The binary representation of the digit. If you represent each digit as a 4-bit binary number (e.g., 0000 for 0, 0001 for 1, ..., 1001 for 9), your input layer will have **4 neurons**. Alternatively, if you represent digits using a fixed-size grid (e.g., 5x5 pixels as binary 0/1 values), your input layer would have **25 neurons**. The prompt suggests "binary value of the digit," which leans towards the 4-bit representation for simplicity in this context.
* **Output**: The corresponding class (0-9). This requires an **output layer with 10 neurons**, one for each digit. Use a **softmax activation function** on the output layer to get probabilities for each class, and the class with the highest probability is the network's prediction.

### Network Structure

* **Input Layer**: 4 neurons (for 4-bit binary input).
* **Hidden Layer(s)**: Experiment with the number of hidden layers and neurons per layer (e.g., one hidden layer with 8-16 neurons).
* **Output Layer**: 10 neurons.

### Learning Error and Testing

Track the **learning error** (e.g., cross-entropy loss or MSE) per epoch. After training, **test your network** by feeding in the binary representations of digits 0 through 9 and verifying if the network correctly identifies them.

---

## Project I: Alphabet Character Recognition

This project involves building a more robust neural network with a graphical interface.

### a) Learning Base Creation

* **Data Representation**: Each character (lowercase and uppercase a-z) needs to be converted into a numerical format. A common approach is to represent characters as small **grayscale or binary images** (e.g., 10x10 or 20x20 pixels). Each pixel then becomes an input feature.
    * You'll need to create images for each character, or draw them directly in your GUI and save them as training examples.
* **Dataset Size**: Aim for multiple variations of each character (different fonts, slight rotations, thicknesses) to improve generalization.

### b) Neural Network Structure

* **Input Layer**: Number of neurons will be `image_width * image_height` (e.g., 10x10 image = 100 input neurons).
* **Hidden Layer(s)**: Experiment with the number of hidden layers and neurons. Given the complexity of characters, you might need more neurons than in previous exercises.
* **Output Layer**: `26 (lowercase) + 26 (uppercase) = 52 neurons`. Or, if you treat 'a' and 'A' as the same class, then 26 neurons. The prompt states "lowercase and uppercase alphabetic characters", so 52 classes are likely.
* **Activation Functions**: Sigmoid or ReLU for hidden layers, Softmax for the output layer.

### c) Gradient Backpropagation Learning

Apply the **backpropagation algorithm** as detailed in Exercise 2. You'll need to define:

* **Loss function**: Categorical cross-entropy is typical for multi-class classification.
* **Optimizer**: Stochastic Gradient Descent (SGD) or Adam are good choices.
* **Learning Rate**: Fine-tune this parameter.

### d) Testing

* **Input Characters**: Allow the user to "draw" a character in the GUI or load a character image.
* **Prediction**: Pass the character's pixel data through the trained network.
* **Output**: Display the network's predicted character. Provide feedback on correctness.

---

## Project II: Robotic Manipulator Control (Readme File)

This project describes a robotic manipulator problem. The request is to write a `readme` file for this exercise.

```markdown
# Robotic Manipulator Control (Two-Link Arm)

## Project Overview

This project addresses the fundamental problem of controlling a **two-link robotic manipulator** in a 2D plane. The manipulator consists of two rigid links connected by two rotoid (revolute) joints. The objective is to determine the necessary joint movements (angles) to move the end-effector (the tip of the second link) from an arbitrary **initial position** to a desired **final position**.

## Problem Description

The core challenge lies in solving the **inverse kinematics problem** for this two-link arm. Given a desired end-effector (x, y) coordinate, we need to calculate the corresponding angles of the two rotoid joints ($\theta_1$ and $\theta_2$).

### Robot Configuration

* **Link 1**: Length $L_1$, connected to a fixed base at its first joint.
* **Link 2**: Length $L_2$, connected to Link 1 at its second joint.
* **Joints**: Two rotoid joints allow rotation in the plane.

### Key Challenges

1.  **Inverse Kinematics**:
    * Unlike forward kinematics (calculating end-effector position from joint angles), inverse kinematics often has multiple solutions, no solutions (unreachable points), or singularities.
    * Developing an algorithm to find the correct joint angles for a given target (x, y) coordinate.
2.  **Path Planning/Trajectory Generation**:
    * How to move the arm smoothly from the initial position to the final position. This involves generating a sequence of intermediate target points or joint angles.
3.  **Control**:
    * How to implement a control mechanism (e.g., PID controller, or a more advanced neural network-based controller) to drive the joints to the desired angles.

## Potential Approaches (to be explored in the project)

Several methods can be employed to solve this problem:

### 1. Analytical Inverse Kinematics

* **Description**: Deriving closed-form mathematical equations to directly calculate joint angles from the end-effector position. This is feasible for simpler manipulators like this two-link arm.
* **Advantages**: Precise, fast computation.
* **Challenges**: Can become complex for manipulators with more degrees of freedom or different joint types. Dealing with multiple solutions.

### 2. Numerical Inverse Kinematics (e.g., Jacobian-based methods)

* **Description**: Iteratively adjusting joint angles to minimize the error between the current end-effector position and the target position. This often involves the **Jacobian matrix**.
* **Advantages**: More generalizable to complex robots.
* **Challenges**: Can be computationally more intensive, sensitive to initial guesses, may get stuck in local minima.

### 3. Neural Network Approach

* **Description**: Training a neural network to learn the mapping from end-effector (x, y) coordinates to joint angles ($\theta_1$, $\theta_2$).
* **Advantages**: Can learn complex non-linear relationships, potentially robust to noise.
* **Challenges**: Requires a large training dataset (forward kinematics can be used to generate data), "black-box" nature, generalization to unseen configurations.

## Project Deliverables (Examples)

* **Code Implementation**: Python (with libraries like NumPy, Matplotlib for visualization), C++, etc.
* **Simulation**: Graphical representation of the robotic arm moving from start to end.
* **Analysis**: Discussion of chosen methods, challenges encountered, and results.
* **Readme File**: This document!

---
```
# Written Number Prediction
https://www.kaggle.com/code/mahmoudwalid1/written-number-prediction?select=train.csv

# MNIST digit recognition
https://www.kaggle.com/code/yadavabhishekz/mnist-digit-recognition

# CNN keras 99%
https://www.kaggle.com/code/nagarajukuruva/cnn-keras-with-99
