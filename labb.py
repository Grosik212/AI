import random

learning_rate = 0.3
weight_1 = random.uniform(-1, 1)
weight_2 = random.uniform(-1, 1)
weight_3 = random.uniform(-1, 1)
steps = ((1, 0, 1), (0, 1, 1), (0, 0, 1), (1, 1, 1))

print(f"Początkowe wagi: w1: {weight_1}, w2: {weight_2}, w3: {weight_3}")

repeat = True
steps_counter = 0

while repeat:
    print("-----------------------")
    repeat = False

    for current_step in steps:
        x1, x2, x3 = current_step
        s = x1 * weight_1 + x2 * weight_2 + x3 * weight_3
        y = 1 if s > 0 else 0
        d = x1 or x2
        delta = d - y

        if delta != 0:
            repeat = True
            weight_1 += learning_rate * delta * x1
            weight_2 += learning_rate * delta * x2
            weight_3 += learning_rate * delta * x3

        print(f"Wejścia: {x1}{x2}{x3}, s: {s}, y: {y}, d: {d}, delta: {delta}")

        steps_counter += 1

print("-----------------------")
print(f"Końcowe wagi: w1: {weight_1}, w2: {weight_2}, w3: {weight_3}")
print(f"Liczba kroków: {steps_counter}")





