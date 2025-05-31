import tensorflow as tf
import matplotlib.pyplot as plt
from keras.saving import save_model


# 1. Load datasets (training and validation)
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    "DATASET",  # Folder containing subfolders of classes
    validation_split=0.2,  # Use 20% of data for validation
    subset="training",
    seed=123,
    image_size=(28, 28),  # Resize all images to 28x28
    batch_size=32,
)

val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    "DATASET",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(28, 28),
    batch_size=32,
)

# 2. Get class names BEFORE mapping
class_names = train_ds_raw.class_names
num_classes = len(class_names)
print(f"Detected Classes: {class_names}")

# 3. Normalize pixel values (0–1)
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
train_ds = train_ds_raw.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds_raw.map(lambda x, y: (normalization_layer(x), y))

# 4. Optimize performance with prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 5. Define CNN model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

# 6. Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# 7. Train the model
history = model.fit(train_ds, validation_data=val_ds, epochs=15)


save_model(model, "my_model.keras")

# 8. Plot training and validation accuracy
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()