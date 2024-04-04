import random
from deap import base, creator, tools, algorithms

# Inicjalizacja problemu genetycznego
creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
creator.create("Individual", list, fitness = creator.FitnessMin)


# Funkcja celu
def calculate_fitness(individual):
    a, b, c, d = individual
    points = [(-5, -150), (-4, -77), (-3, -30), (-2, 0), (-1, 10), (1 / 2, 131 / 8), (1, 18), (2, 25), (3, 32), (4, 75),
              (5, 130)]
    error = sum((a * x ** 3 + b * x ** 2 + c * x + d - y) ** 2 for x, y in points)
    return error,


# Ograniczenia dla współczynników
def check_bounds(min_value, max_value):
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for child in offspring:
                for i in range(len(child)):
                    child[i] = min_value if child[i] < min_value else max_value if child[i] > max_value else child[i]
            return offspring

        return wrapper

    return decorator


def round_coefficients(individual):
    return [round(coeff) for coeff in individual]


# Algorytm genetyczny
def genetic_algorithm(pop_size=100, n_gen=50, cx_prob=0.7, mut_prob=0.2):
    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, -15, 15)  # Ograniczenie zakresu dla współczynników
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n = 4)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", calculate_fitness)
    toolbox.register("mate", tools.cxBlend, alpha = 0.5)
    toolbox.register("mutate", tools.mutUniformInt, low = -15, up = 15, indpb = 0.2)
    toolbox.decorate("mate", check_bounds(-15, 15))  # Ograniczenie zakresu dla krzyżowania
    toolbox.decorate("mutate", check_bounds(-15, 15))  # Ograniczenie zakresu dla mutacji
    toolbox.register("select", tools.selTournament, tournsize = 3)

    population = toolbox.population(n = pop_size)
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    for gen in range(n_gen):
        offspring = algorithms.varAnd(population, toolbox, cxpb = cx_prob, mutpb = mut_prob)
        fits = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
        population = toolbox.select (offspring + population, k = pop_size)

        best_ind = tools.selBest(population, k = 1)[0]
        best_ind_rounded = round_coefficients(best_ind)

        print(f"Generacja {gen + 1}: Najlepsza wartość funkcji celu = {best_ind.fitness.values [0]}")
        print(f"Współczynniki (zaokrąglone do liczb całkowitych): {best_ind_rounded}")

    return best_ind


best_solution = genetic_algorithm ()
print("Najlepsze współczynniki (a, b, c, d):", best_solution)
print("Wartość funkcji celu dla najlepszego osobnika:", best_solution.fitness.values[0])
