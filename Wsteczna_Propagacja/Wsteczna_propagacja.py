import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def mean_squared_error(predictions, targets):
    return np.mean((predictions - targets) ** 2)


def train_xor_neural_network(inputs, targets, epochs, learning_rate):

    input_layer_neurons = inputs.shape[1]
    hidden_layer_neurons = 4
    output_layer_neurons = 1

    hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
    output_weights = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))

    print("\nWyniki przed treningiem:")
    hidden_layer_input = np.dot(inputs, hidden_weights)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, output_weights)
    predicted_output_before_training = sigmoid(output_layer_input)

    for i in range(len(inputs)):
        print(f"Wejścia: {inputs[i]}, Przewidywane: {predicted_output_before_training[i][0]:.4f}, Oczekiwane: {targets[i][0]}")

    for epoch in range(epochs):
        hidden_layer_input = np.dot(inputs, hidden_weights)
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, output_weights)
        predicted_output = sigmoid(output_layer_input)

        error = targets - predicted_output

        output_error = error * sigmoid_derivative(predicted_output)
        hidden_layer_error = output_error.dot(output_weights.T) * sigmoid_derivative(hidden_layer_output)

        output_weights += hidden_layer_output.T.dot(output_error) * learning_rate
        hidden_weights += inputs.T.dot(hidden_layer_error) * learning_rate

        if epoch % 1000 == 0:
            mse = mean_squared_error(predicted_output, targets)
            print(f'Epoch: {epoch}, Mean Squared Error: {mse}')

    print("\nUczenie zakończone.")
    print("Wagi warstwy ukrytej:")
    print(hidden_weights)
    print("Wagi warstwy wyjściowej:")
    print(output_weights)

    print("\nWyniki po treningu:")
    final_hidden_layer_input = np.dot(inputs, hidden_weights)
    final_hidden_layer_output = sigmoid(final_hidden_layer_input)
    final_output_layer_input = np.dot(final_hidden_layer_output, output_weights)
    final_predicted_output = sigmoid(final_output_layer_input)

    for i in range(len(inputs)):
        print(f"Wejścia: {inputs[i]}, Przewidywane: {final_predicted_output[i][0]:.4f}, Oczekiwane: {targets[i][0]}")


inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

targets = np.array([[0],
                    [1],
                    [1],
                    [0]])

train_xor_neural_network(inputs, targets, epochs=50000, learning_rate=0.5)
