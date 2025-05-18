import numpy as np


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# One-hot encoding of digits 0–9
def one_hot_encode(n, size=10):
    vec = np.zeros(size)
    vec[n] = 1
    return vec


# Binary input for digits 0 to 9
X = np.array(
    [
        [0, 0, 0, 0],  # 0
        [0, 0, 0, 1],  # 1
        [0, 0, 1, 0],  # 2
        [0, 0, 1, 1],  # 3
        [0, 1, 0, 0],  # 4
        [0, 1, 0, 1],  # 5
        [0, 1, 1, 0],  # 6
        [0, 1, 1, 1],  # 7
        [1, 0, 0, 0],  # 8
        [1, 0, 0, 1],  # 9
    ]
)

# Expected output (one-hot vectors)
y = np.array([one_hot_encode(i) for i in range(10)])

# Network parameters
np.random.seed(42)
input_size = 4
hidden1_size = 4
hidden2_size = 4
output_size = 10

# Weights and biases
W1 = np.random.uniform(-1, 1, (input_size, hidden1_size))
b1 = np.zeros((1, hidden1_size))

W2 = np.random.uniform(-1, 1, (hidden1_size, hidden2_size))
b2 = np.zeros((1, hidden2_size))

W3 = np.random.uniform(-1, 1, (hidden2_size, output_size))
b3 = np.zeros((1, output_size))

# Training parameters
epochs = 10000
lr = 0.1

print("Epoch\tLoss")

# Training loop
for epoch in range(1, epochs + 1):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(a2, W3) + b3
    a3 = sigmoid(z3)

    # Compute loss
    error = y - a3
    loss = np.mean(np.square(error))

    if epoch % 1000 == 0:
        print(f"{epoch}\t{loss:.6f}")

    # Backpropagation
    d3 = error * sigmoid_derivative(a3)
    d2 = np.dot(d3, W3.T) * sigmoid_derivative(a2)
    d1 = np.dot(d2, W2.T) * sigmoid_derivative(a1)

    # Update weights and biases
    W3 += lr * np.dot(a2.T, d3)
    b3 += lr * np.sum(d3, axis=0, keepdims=True)

    W2 += lr * np.dot(a1.T, d2)
    b2 += lr * np.sum(d2, axis=0, keepdims=True)

    W1 += lr * np.dot(X.T, d1)
    b1 += lr * np.sum(d1, axis=0, keepdims=True)

# Final predictions
print("\nPredictions after training:")
for i, input_vector in enumerate(X):
    a1 = sigmoid(np.dot(input_vector, W1) + b1)
    a2 = sigmoid(np.dot(a1, W2) + b2)
    a3 = sigmoid(np.dot(a2, W3) + b3)
    predicted = np.argmax(a3)
    print(f"Input: {input_vector}, Predicted: {predicted}, Actual: {i}")