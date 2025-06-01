# gui_app.py

import numpy as np
import tkinter as tk
from tkinter import messagebox


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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

        tk.Label(
            self.root, text="Alphabet Character Recognizer", font=("Arial", 16)
        ).pack(pady=10)

        tk.Label(self.root, text="Enter a character (a-z or A-Z):").pack(pady=5)
        self.input_entry = tk.Entry(
            self.root, font=("Arial", 14), width=5, justify="center"
        )
        self.input_entry.pack(pady=5)

        self.prediction_label = tk.Label(
            self.root, text="Prediction will appear here", font=("Arial", 14)
        )
        self.prediction_label.pack(pady=20)

        self.confidence_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.confidence_label.pack(pady=5)

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


def load_model_and_run_gui():
    data = np.load("model_weights.npz")
    W1 = data["W1"]
    b1 = data["b1"]
    W2 = data["W2"]
    b2 = data["b2"]

    with open("characters.txt", "r") as f:
        characters = list(f.read())

    root = tk.Tk()
    CharacterRecognizerUI(root, W1, b1, W2, b2, characters)
    root.mainloop()


if __name__ == "__main__":
    load_model_and_run_gui()
