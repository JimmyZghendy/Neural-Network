import numpy as np
import string


# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Create the alphabet dataset
def create_alphabet_dataset():
    # Create a list of all lowercase and uppercase letters
    lowercase = list(string.ascii_lowercase)
    uppercase = list(string.ascii_uppercase)
    all_chars = lowercase + uppercase

    # Create input vectors (one-hot encoded)
    X = np.eye(52)

    # Create output vectors (same as input for auto-associative learning)
    y = np.eye(52)

    return X, y, all_chars


# Create the dataset
X, y, characters = create_alphabet_dataset()

# Initialize weights and biases
np.random.seed(42)
input_size = 52
hidden_size = 64
output_size = 52

# Initialize weights with Xavier/Glorot initialization
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
b2 = np.zeros((1, output_size))

# Training parameters
epochs = 100000
lr = 0.05
print_interval = 2000

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

    # Print loss periodically
    if epoch % print_interval == 0 or epoch == 1:
        print(f"{epoch}\t{loss:.6f}")

    # Backward pass
    d_output = error * sigmoid_derivative(a2)
    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(a1)

    # Update weights and biases
    W2 += lr * np.dot(a1.T, d_output)
    b2 += lr * np.sum(d_output, axis=0, keepdims=True)

    W1 += lr * np.dot(X.T, d_hidden)
    b1 += lr * np.sum(d_hidden, axis=0, keepdims=True)


# Test the network
def predict_character(input_vector):
    z1 = np.dot(input_vector, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    return a2


print("\nTesting the network:")
for i in range(52):
    # Get the input vector
    input_vector = X[i : i + 1]  # Keep it as 2D array

    # Get the prediction
    prediction = predict_character(input_vector)

    # Find the predicted character index
    predicted_index = np.argmax(prediction)

    # Get the actual and predicted characters
    actual_char = characters[i]
    predicted_char = characters[predicted_index]

    # Print the result
    print(
        f"Input: {actual_char}, Predicted: {predicted_char}, Correct: {actual_char == predicted_char}"
    )

    # Also print the confidence for the correct prediction
    confidence = prediction[0][i]
    print(f"Confidence for correct character: {confidence:.4f}")
