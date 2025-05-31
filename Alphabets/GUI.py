import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model


# Load your trained model (make sure you've saved it, or load from the code session)
# If you haven't saved, save your model first:
model = load_model("my_model.keras")

# Assuming you have class_names available:
class_names = model.layers[-1].get_config()["units"]
# Actually, better to save class_names externally or hardcode here:
# For example:
class_names = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "Q",
    "R",
    "T",
    "W",
    "Y",
    "i",
    "o",
    "p",
    "s",
    "u",
    "v",
    "x",
    "z",
]


def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array[np.newaxis, ...]  # Add batch dimension
    return img_array


def predict():
    if not file_path.get():
        result_label.config(text="Please select an image first.")
        return

    img_array = preprocess_image(file_path.get())
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    result_label.config(text=f"Prediction: {predicted_class} ({confidence * 100:.2f}%)")


def open_file_dialog():
    filename = filedialog.askopenfilename(
        filetypes=[
            ("PNG Images", "*.png"),
            ("JPEG Images", ".jpg;.jpeg"),
            ("All files", "."),
        ]
    )
    if filename:
        file_path.set(filename)
        img = Image.open(filename).resize((150, 150))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        result_label.config(text="")


# Setup GUI
root = tk.Tk()
root.title("Alphabet Classifier")

file_path = tk.StringVar()

btn_browse = Button(root, text="Select Image", command=open_file_dialog)
btn_browse.pack(pady=10)

image_label = Label(root)
image_label.pack()

btn_predict = Button(root, text="Predict", command=predict)
btn_predict.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 16))
result_label.pack(pady=10)

root.mainloop()