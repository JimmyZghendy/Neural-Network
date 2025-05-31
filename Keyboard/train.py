import numpy as np
import string
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def create_alphabet_dataset():
    lowercase = list(string.ascii_lowercase)
    uppercase = list(string.ascii_uppercase)
    all_chars = lowercase + uppercase
    X = np.eye(52)
    y = np.eye(52)
    return X, y, all_chars


def train_neural_network():
    X, y, characters = create_alphabet_dataset()

    # Shuffle and split into train and validation sets (80% train, 20% val)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(0.8 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    np.random.seed(42)
    input_size = 52
    hidden_size = 64
    output_size = 52

    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((1, output_size))

    epochs = 100000
    lr = 0.05
    print_interval = 2000

    train_losses = []
    val_losses = []

    print("Training neural network...")
    print("Epoch\tTrain Loss\tVal Loss")

    for epoch in range(1, epochs + 1):
        # ---- Forward pass (train)
        z1 = np.dot(X_train, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        # ---- Forward pass (val)
        z1_val = np.dot(X_val, W1) + b1
        a1_val = sigmoid(z1_val)
        z2_val = np.dot(a1_val, W2) + b2
        a2_val = sigmoid(z2_val)

        # ---- Loss
        train_loss = np.mean((y_train - a2) ** 2)
        val_loss = np.mean((y_val - a2_val) ** 2)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % print_interval == 0 or epoch == 1:
            print(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}")

        # ---- Backward pass
        error = y_train - a2
        d_output = error * sigmoid_derivative(a2)
        d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(a1)

        W2 += lr * np.dot(a1.T, d_output)
        b2 += lr * np.sum(d_output, axis=0, keepdims=True)
        W1 += lr * np.dot(X_train.T, d_hidden)
        b1 += lr * np.sum(d_hidden, axis=0, keepdims=True)

    print("\nTraining complete!")

    # Save model
    np.savez("model_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)
    with open("characters.txt", "w") as f:
        f.write("".join(characters))

    # ---- Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_neural_network()
