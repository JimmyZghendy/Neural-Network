import numpy as np


# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# XOR input and expected output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

y = np.array([[0], [1], [1], [0]])

# Initialize weights and biases
np.random.seed(42)
input_size = 2
hidden_size = 2
output_size = 1

W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
b1 = np.zeros((1, hidden_size))

W2 = np.random.uniform(-1, 1, (hidden_size, output_size))
b2 = np.zeros((1, output_size))


# Training parameters
epochs = 10000
lr = 0.1

print("Epoch\tError")

# Training loop
for epoch in range(1, epochs + 1):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Compute error
    error = y - a2
    loss = np.mean(np.square(error))

    # Print loss per cycle
    print(f"{epoch}\t{loss:.6f}")

    # Backward pass
    d_output = error * sigmoid_derivative(a2)
    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(a1)

    # Update weights and biases
    W2 += lr * np.dot(a1.T, d_output)
    b2 += lr * np.sum(d_output, axis=0, keepdims=True)

    W1 += lr * np.dot(X.T, d_hidden)
    b1 += lr * np.sum(d_hidden, axis=0, keepdims=True)

# Final test after training
print("\nFinal predictions after training:")
for i, input_vector in enumerate(X):
    z1 = np.dot(input_vector, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    print(f"Input: {input_vector}, Predicted: {round(a2[0][0])}, Actual: {y[i][0]}")