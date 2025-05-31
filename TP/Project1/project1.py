import numpy as np
import string
import matplotlib.pyplot as plt

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

# Function to calculate accuracy
def calculate_accuracy(X, y, W1, b1, W2, b2):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    
    # Get predictions
    predictions = np.argmax(a2, axis=1)
    actual = np.argmax(y, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == actual)
    return accuracy

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
epochs = 1000
lr = 0.05
print_interval = 2000

# Lists to store metrics for plotting
losses = []
accuracies = []
epoch_list = []

print("Epoch\tLoss\t\tAccuracy")

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
    
    # Calculate accuracy
    accuracy = calculate_accuracy(X, y, W1, b1, W2, b2)
    
    # Print loss and accuracy periodically
    if epoch % print_interval == 0 or epoch == 1:
        losses.append(loss)
        accuracies.append(accuracy)
        epoch_list.append(epoch)
        print(f"{epoch}\t{loss:.6f}\t{accuracy:.4f}")
    
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

# Collect testing results for visualization
test_results = []
confidence_scores = []
correct_predictions = []

print("\nTesting the network:")
for i in range(52):
    # Get the input vector
    input_vector = X[i:i+1]  # Keep it as 2D array
    
    # Get the prediction
    prediction = predict_character(input_vector)
    
    # Find the predicted character index
    predicted_index = np.argmax(prediction)
    
    # Get the actual and predicted characters
    actual_char = characters[i]
    predicted_char = characters[predicted_index]
    
    # Check if prediction is correct
    is_correct = actual_char == predicted_char
    correct_predictions.append(is_correct)
    
    # Get confidence for the correct prediction
    confidence = prediction[0][i]
    confidence_scores.append(confidence)
    
    # Store result
    test_results.append({
        'actual': actual_char,
        'predicted': predicted_char,
        'correct': is_correct,
        'confidence': confidence
    })
    
    # Print the result
    print(f"Input: {actual_char}, Predicted: {predicted_char}, Correct: {is_correct}")
    print(f"Confidence for correct character: {confidence:.4f}")

# Create visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Training Loss Graph
ax1.plot(epoch_list, losses, 'b-', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss Over Time')
ax1.grid(True, alpha=0.3)

# 2. Training Accuracy Graph
ax2.plot(epoch_list, accuracies, 'g-', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training Accuracy Over Time')
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)

# 3. Testing Results - Correct vs Incorrect Predictions
correct_count = sum(correct_predictions)
incorrect_count = len(correct_predictions) - correct_count
labels = ['Correct', 'Incorrect']
sizes = [correct_count, incorrect_count]
colors = ['#2ecc71', '#e74c3c']

ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax3.set_title(f'Testing Results\n({correct_count}/{len(correct_predictions)} correct)')

# 4. Confidence Scores Distribution
ax4.hist(confidence_scores, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Confidence Score')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of Confidence Scores')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final statistics
print(f"\nFinal Training Statistics:")
print(f"Final Loss: {losses[-1]:.6f}")
print(f"Final Accuracy: {accuracies[-1]:.4f}")
print(f"Testing Accuracy: {correct_count}/{len(correct_predictions)} = {correct_count/len(correct_predictions):.4f}")
print(f"Average Confidence Score: {np.mean(confidence_scores):.4f}")
print(f"Min Confidence Score: {np.min(confidence_scores):.4f}")
print(f"Max Confidence Score: {np.max(confidence_scores):.4f}")