import numpy as np
import string
import tkinter as tk
from tkinter import messagebox


# Neural Network Implementation
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

    print("Training neural network...")
    print("Epoch\tError")

    for epoch in range(1, epochs + 1):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        # Compute error
        error = y - a2
        loss = np.mean(np.square(error))

        if epoch % print_interval == 0 or epoch == 1:
            print(f"{epoch}\t{loss:.6f}")

        # Backward pass
        d_output = error * sigmoid_derivative(a2)
        d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(a1)

        # Update weights
        W2 += lr * np.dot(a1.T, d_output)
        b2 += lr * np.sum(d_output, axis=0, keepdims=True)
        W1 += lr * np.dot(X.T, d_hidden)
        b1 += lr * np.sum(d_hidden, axis=0, keepdims=True)

    print("\nTraining complete!")
    return W1, b1, W2, b2, characters


# UI Implementation
class CharacterRecognizerUI:
    def __init__(self, root, W1, b1, W2, b2, characters):
        self.root = root
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.characters = characters
        self.char_to_index = {char: idx for idx, char in enumerate(characters)}

        self.setup_ui()

    def setup_ui(self):
        self.root.title("Alphabet Character Recognizer")
        self.root.geometry("400x300")

        # Title
        tk.Label(
            self.root, text="Alphabet Character Recognizer", font=("Arial", 16)
        ).pack(pady=10)

        # Input label and field
        tk.Label(self.root, text="Enter a character (a-z or A-Z):").pack(pady=5)
        self.input_entry = tk.Entry(
            self.root, font=("Arial", 14), width=5, justify="center"
        )
        self.input_entry.pack(pady=5)

        # Prediction display
        self.prediction_label = tk.Label(
            self.root, text="Prediction will appear here", font=("Arial", 14)
        )
        self.prediction_label.pack(pady=20)

        # Confidence display
        self.confidence_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.confidence_label.pack(pady=5)

        # Predict button
        tk.Button(self.root, text="Predict", command=self.predict_character).pack(
            pady=10
        )

    def predict_character(self):
        input_char = self.input_entry.get()

        if len(input_char) != 1 or input_char not in self.char_to_index:
            messagebox.showerror(
                "Invalid Input",
                "Please enter a single alphabetic character (a-z or A-Z)",
            )
            return

        char_index = self.char_to_index[input_char]
        input_vector = np.zeros((1, 52))
        input_vector[0, char_index] = 1

        # Neural network forward pass
        z1 = np.dot(input_vector, self.W1) + self.b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        prediction = sigmoid(z2)

        predicted_index = np.argmax(prediction)
        predicted_char = self.characters[predicted_index]
        confidence = prediction[0, char_index]

        self.prediction_label.config(text=f"Predicted: {predicted_char}")

        if input_char == predicted_char:
            self.prediction_label.config(fg="green")
            self.confidence_label.config(
                text=f"Confidence: {confidence:.4f}", fg="green"
            )
        else:
            self.prediction_label.config(fg="red")
            self.confidence_label.config(
                text=f"Confidence: {confidence:.4f} (Incorrect)", fg="red"
            )


# Main execution
if __name__ == "__main__":
    # Train the neural network
    W1, b1, W2, b2, characters = train_neural_network()

    # Create and run the UI
    root = tk.Tk()
    app = CharacterRecognizerUI(root, W1, b1, W2, b2, characters)
    root.mainloop()
