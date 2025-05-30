import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# Forward Kinematics (to generate training data)
def forward_kinematics(theta1, theta2, a1=3, a2=2):
    x = a1 * np.cos(theta1) + a2 * np.cos(theta1 + theta2)
    y = a1 * np.sin(theta1) + a2 * np.sin(theta1 + theta2)
    return x, y


# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Neural Network for Inverse Kinematics
class InverseKinematicsNN:
    def __init__(
        self, input_size=2, hidden_size=10, output_size=4
    ):  # Changed output to 4
        # Xavier/Glorot initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2  # Linear activation for output (regression)
        return self.a2

    def backward(self, X, y, lr=0.001):  # Reduced learning rate
        m = X.shape[0]

        # Output layer error
        error = self.a2 - y
        d_output = error

        # Hidden layer error
        d_hidden = np.dot(d_output, self.W2.T) * sigmoid_derivative(self.a1)

        # Update weights and biases
        self.W2 -= lr * np.dot(self.a1.T, d_output) / m
        self.b2 -= lr * np.sum(d_output, axis=0, keepdims=True) / m
        self.W1 -= lr * np.dot(X.T, d_hidden) / m
        self.b1 -= lr * np.sum(d_hidden, axis=0, keepdims=True) / m

        return np.mean(np.square(error))


# Generate training data
def generate_training_data(num_samples=1000):
    theta1 = np.random.uniform(0, 2 * np.pi, num_samples)
    theta2 = np.random.uniform(-np.pi, np.pi, num_samples)

    X = np.zeros((num_samples, 2))
    y = np.zeros((num_samples, 4))  # Now outputs 4 values

    for i in range(num_samples):
        x, y_pos = forward_kinematics(theta1[i], theta2[i])
        X[i] = [x, y_pos]
        # Predict sin/cos of angles instead of angles directly
        y[i] = [
            np.sin(theta1[i]),
            np.cos(theta1[i]),
            np.sin(theta2[i]),
            np.cos(theta2[i]),
        ]

    return X, y


# Normalize data
def normalize(X, y):
    X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
    # Don't normalize y since sin/cos are already in [-1,1]
    return (X - X_mean) / X_std, y, (X_mean, X_std)


# Convert predicted sin/cos back to angles
def sin_cos_to_angle(sin_val, cos_val):
    angle = np.arctan2(sin_val, cos_val)
    return angle if angle >= 0 else angle + 2 * np.pi


# GUI Application
class RobotArmApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Inverse Kinematics")

        # Create neural network
        self.nn = InverseKinematicsNN()

        # Generate and normalize training data
        self.X, self.y = generate_training_data()
        self.X_norm, self.y_norm, self.norm_params = normalize(self.X, self.y)

        # GUI Elements
        self.setup_gui()

        # Plot setup
        self.setup_plots()

    def setup_gui(self):
        # Control Frame
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Training Controls
        ttk.Label(control_frame, text="Epochs:").pack()
        self.epochs_entry = ttk.Entry(control_frame)
        self.epochs_entry.insert(0, "1000")
        self.epochs_entry.pack()

        ttk.Label(control_frame, text="Learning Rate:").pack()
        self.lr_entry = ttk.Entry(control_frame)
        self.lr_entry.insert(0, "0.001")  # Reduced default learning rate
        self.lr_entry.pack()

        self.train_button = ttk.Button(
            control_frame, text="Train Network", command=self.train_network
        )
        self.train_button.pack(pady=10)

        # Test Controls
        ttk.Label(control_frame, text="Test X:").pack()
        self.test_x_entry = ttk.Entry(control_frame)
        self.test_x_entry.insert(0, "2.0")
        self.test_x_entry.pack()

        ttk.Label(control_frame, text="Test Y:").pack()
        self.test_y_entry = ttk.Entry(control_frame)
        self.test_y_entry.insert(0, "2.0")
        self.test_y_entry.pack()

        self.test_button = ttk.Button(
            control_frame, text="Test Position", command=self.test_position
        )
        self.test_button.pack(pady=10)

        # Status
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(control_frame, textvariable=self.status_var).pack()

    def setup_plots(self):
        # Plot Frame
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)

        # Initial plots
        self.ax1.set_title("Workspace")
        self.ax1.scatter(self.X[:, 0], self.X[:, 1], s=1, alpha=0.3)
        self.ax1.set_xlabel("X")
        self.ax1.set_ylabel("Y")
        self.ax1.grid(True)

        self.ax2.set_title("Robotic Arm")
        self.ax2.set_xlim(-6, 6)
        self.ax2.set_ylim(-6, 6)
        self.ax2.grid(True)

        # Add to Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def train_network(self):
        try:
            epochs = int(self.epochs_entry.get())
            lr = float(self.lr_entry.get())
        except ValueError:
            self.status_var.set("Invalid input values")
            return

        self.status_var.set("Training...")
        self.root.update()

        losses = []
        for epoch in range(epochs):
            # Forward and backward pass
            self.nn.forward(self.X_norm)
            loss = self.nn.backward(self.X_norm, self.y_norm, lr)
            losses.append(loss)

            if epoch % 100 == 0:
                self.status_var.set(
                    f"Training... Epoch {epoch}/{epochs}, Loss: {loss:.4f}"
                )
                self.root.update()

        # Plot training loss
        self.ax1.clear()
        self.ax1.set_title("Training Loss")
        self.ax1.plot(losses)
        self.ax1.set_xlabel("Epoch")
        self.ax1.set_ylabel("MSE Loss")
        self.ax1.grid(True)

        self.canvas.draw()
        self.status_var.set(f"Training complete. Final loss: {loss:.4f}")

    def test_position(self):
        try:
            x = float(self.test_x_entry.get())
            y = float(self.test_y_entry.get())
        except ValueError:
            self.status_var.set("Invalid position values")
            return

        # Normalize input
        X_mean, X_std = self.norm_params
        x_norm = (x - X_mean[0]) / X_std[0]
        y_norm = (y - X_mean[1]) / X_std[1]

        # Predict sin/cos values
        pred_norm = self.nn.forward(np.array([[x_norm, y_norm]]))

        # Convert predicted sin/cos back to angles
        sin_theta1, cos_theta1, sin_theta2, cos_theta2 = pred_norm[0]
        theta1_pred = sin_cos_to_angle(sin_theta1, cos_theta1)
        theta2_pred = sin_cos_to_angle(sin_theta2, cos_theta2)

        # Calculate forward kinematics to check
        x_pred, y_pred = forward_kinematics(theta1_pred, theta2_pred)

        # Update status
        self.status_var.set(
            f"Input: ({x:.2f}, {y:.2f})\n"
            f"Predicted angles: θ1={theta1_pred:.2f} rad, θ2={theta2_pred:.2f} rad\n"
            f"Reconstructed position: ({x_pred:.2f}, {y_pred:.2f})"
        )

        # Plot arm configuration
        self.plot_arm(theta1_pred, theta2_pred)

    def plot_arm(self, theta1, theta2):
        a1, a2 = 3, 2
        x1 = a1 * np.cos(theta1)
        y1 = a1 * np.sin(theta1)
        x2 = x1 + a2 * np.cos(theta1 + theta2)
        y2 = y1 + a2 * np.sin(theta1 + theta2)

        self.ax2.clear()
        self.ax2.set_title("Robotic Arm Configuration")
        self.ax2.plot([0, x1, x2], [0, y1, y2], "ro-", linewidth=2, markersize=8)
        self.ax2.plot(0, 0, "ks", markersize=10)  # Base
        self.ax2.set_xlim(-6, 6)
        self.ax2.set_ylim(-6, 6)
        self.ax2.grid(True)
        self.ax2.set_aspect("equal")

        self.canvas.draw()


# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = RobotArmApp(root)
    root.mainloop()
