class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # Initialize weights with zeros (including bias term)
        self.weights = [0.0] * (input_size + 1)  # [bias, w1, w2]
        self.learning_rate = learning_rate

    def predict(self, inputs):
        # Start with bias term
        summation = self.weights[0]
        # Add weighted inputs
        for i in range(len(inputs)):
            summation += self.weights[i + 1] * inputs[i]
        # Apply step function
        return 1 if summation >= 0 else 0

    def train(self, training_data, labels, epochs=10):
        errors_per_epoch = []

        for epoch in range(epochs):
            total_error = 0
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                total_error += abs(error)  # Count errors

                # Update bias (weight[0])
                self.weights[0] += (
                    self.learning_rate * error * 1
                )  # Input for bias is always 1

                # Update other weights
                for i in range(len(inputs)):
                    self.weights[i + 1] += self.learning_rate * error * inputs[i]

            errors_per_epoch.append(total_error)
            print(f"Epoch {epoch + 1}, Error: {total_error}")

            # Early stopping if no errors
            if total_error == 0:
                break

        return errors_per_epoch


# OR problem training data
training_data = [[0, 0], [0, 1], [1, 0], [1, 1]]

# OR problem labels (0 for false, 1 for true)
labels = [0, 1, 1, 1]

# Create and train the perceptron
perceptron = Perceptron(input_size=2, learning_rate=0.1)
errors = perceptron.train(training_data, labels, epochs=10)

# Test the trained perceptron
print("\nTesting the trained perceptron:")
for inputs in training_data:
    prediction = perceptron.predict(inputs)
    print(f"Input: {inputs}, OR Result: {prediction}")

# Final weights
print("\nFinal weights (bias, w1, w2):", perceptron.weights)