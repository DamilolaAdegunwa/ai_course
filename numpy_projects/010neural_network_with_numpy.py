import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        # Forward pass
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = self.softmax(self.output_layer_input)
        return self.output

    def backward(self, X, y, learning_rate):
        # Backward pass
        output_error = self.output - y
        output_gradient = output_error

        hidden_error = np.dot(output_gradient, self.weights_hidden_output.T)
        hidden_gradient = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)

        # Update weights and biases
        self.weights_hidden_output -= learning_rate * np.dot(self.hidden_layer_output.T, output_gradient)
        self.bias_output -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)

        self.weights_input_hidden -= learning_rate * np.dot(X.T, hidden_gradient)
        self.bias_hidden -= learning_rate * np.sum(hidden_gradient, axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            if epoch % 100 == 0:
                loss = -np.mean(np.sum(y * np.log(self.output + 1e-8), axis=1))
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)


# Example input data
if __name__ == "__main__":
    # Dummy dataset (4 samples, 2 features each)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [1, 1], [1, 1], [1, 0], [1, 0]])  # XOR problem
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1]])  # One-hot encoded targets

    # Initialize the neural network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=2)

    # Train the neural network
    nn.train(X, y, epochs=1000, learning_rate=0.1)

    # Test the neural network
    predictions = nn.predict(X)
    print("Predictions:", predictions)
