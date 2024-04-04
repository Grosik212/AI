def hopfield_network(input_vector):
    matrix_size = len(input_vector)
    weights_matrix = []

    for i in range(matrix_size):
        row = []
        for j in range(matrix_size):
            if i == j:
                row.append(0)
            else:
                row.append((2 * input_vector[i] - 1) * (2 * input_vector[j] - 1))
        weights_matrix.append(row)

    return weights_matrix


def predict_pattern(weights_matrix, test_vector):
    result = []

    for k in range(len(weights_matrix)):
        temp_result = sum(weights_matrix[k][l] * test_vector[l] for l in range(len(test_vector)))
        result.append (temp_result)

    predicted_vector = [1 if val > 0 else 0 if val == 0 else test_vector[i] for i, val in enumerate(result)]

    return predicted_vector


input_vector = [1, 1, 1, 1, 1, 1, 1, 1,
                1, 0, 0, 0, 0, 0, 0, 1,
                1, 0, 0, 0, 0, 0, 0, 1,
                1, 0, 0, 0, 0, 0, 0, 1,
                1, 0, 0, 0, 0, 0, 0, 1,
                1, 0, 0, 0, 0, 0, 0, 1,
                1, 0, 0, 0, 0, 0, 0, 1,
                1, 1, 1, 1, 1, 1, 1, 1]

weights = hopfield_network(input_vector)

test_vector = [1, 0, 0, 0, 0, 0, 0, 1,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               1, 0, 0, 0, 0, 0, 0, 1]

predicted_pattern = predict_pattern(weights, test_vector)

print("Wynik:")
for i in range(64):
    if i % 8 == 0 and i > 0:
        print()
    print(predicted_pattern[i], end = " ")
