import random
import sys
from random import getrandbits

import numpy as np

MUTATION_PROBABILITY = 0.2


def magic_d(weights, values, capacity):
    items = len(values)
    mdim = np.zeros([capacity + 1, 2])

    # If this takes too long use dragon_ball from Namek; they are more powerfull.
    for dragon_ball in range(0, len(weights)):
        for cap in range(1, capacity + 1):
            if weights[dragon_ball] <= cap:
                mdim[cap, 1] = max(
                    mdim[cap, 0],
                    values[dragon_ball] + mdim[cap - weights[dragon_ball], 0],
                )
            else:
                mdim[cap, 1] = mdim[cap, 0]
        mdim[:, 0] = mdim[:, 1]
    value = mdim[capacity, 1]
    return value


def init_population(n, m):
    """
    Randomly initialize the population.
    :param n: Number of individuals.
    :param m: Number of chromosomes per individual.
    :return: An n x m matrix with the population individuals.
    """
    population = np.zeros((n, m), dtype=np.uint8)
    for i in range(n):
        x = getrandbits(m)
        print([int(s) for s in f"{x:0{m}b}"])
        population[i] = np.array([int(s) for s in f"{x:0{m}b}"], dtype=np.uint8)

    return population


def get_fitting_value(population, items_values, items_weights, capacity):
    fitting_values = np.zeros(population.shape[0])
    for i, individual in enumerate(population):
        # If the individual exceeds the capacity its value is 0.
        # Otherwise, the value is the total value of the items it selected.
        if items_weights[individual == 1].sum() > capacity:
            fitting_values[i] = 0
        else:
            fitting_values[i] = items_values[individual == 1].sum()

    return fitting_values


def elitism(population, fitting_values, keep):
    return population[np.argpartition(fitting_values, -keep)[-keep:]]


def tournament_selection(k, population, fitting_values, keep):
    selection = np.random.choice(range(population.shape[0]), k)
    selection_val = fitting_values[selection]

    return elitism(population, selection_val, keep)


def mutate(individual):
    for i in range(individual.shape[0]):
        if random.random() <= MUTATION_PROBABILITY:
            individual[i] = 1 ^ individual[i]


def one_point_crossover(parent1, parent2):
    point = np.random.randint(1, parent1.shape[0] - 1)
    offspring1 = np.concatenate((parent1[:point], parent2[point:]))
    offspring2 = np.concatenate((parent2[:point], parent1[point:]))
    return mutate(offspring1), mutate(offspring2)


def two_point_crossover(parent1, parent2):
    point1 = np.random.randint(1, parent1.shape[0] - 1)

    while True:
        point2 = np.random.randint(1, parent1.shape[0] - 1)
        if point2 != point1:
            break

    parent1_s1, parent1_s2, parent1_s3 = np.split(parent1, np.sort([point1, point2]))
    parent2_s1, parent2_s2, parent2_s3 = np.split(parent2, np.sort([point1, point2]))
    offspring1 = np.concatenate((parent1_s1, parent2_s2, parent1_s3))
    offspring2 = np.concatenate((parent2_s1, parent1_s2, parent2_s3))
    return mutate(offspring1), mutate(offspring2)


def ux_crossover(parent1, parent2):
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()
    for i in range(parent1.shape[0]):
        if random.random() < 0.5:
            offspring1[i] = parent2[i]
            offspring2[i] = parent1[i]

    return mutate(offspring1), mutate(offspring2)


def crossover(parent1, parent2, method="one-point"):
    if method == "one-point":
        return one_point_crossover(parent1, parent2)
    elif method == "two-point":
        return two_point_crossover(parent1, parent2)
    elif method == "ux":
        return ux_crossover(parent1, parent2)


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split("\n")

    first_line = lines[0].split()
    items = int(first_line[0])
    capacity = int(first_line[1])

    values = []
    weights = []

    for i in range(1, items + 1):
        line = lines[i]
        parts = line.split()

        values.append(int(parts[0]))
        weights.append(int(parts[1]))

    items = len(values)

    # weights is a list containing the different weights for the items
    # values is a list containing the different values for the items

    # WRITE YOUR OWN CODE HERE #####################################

    ## MAGIC ##
    # best_value = magic_d(weights,values,capacity)

    value = 0
    taken = items * [0]

    value = magic_d(weights, values, capacity)

    # STOP WRITING YOUR CODE HERE ###################################

    output_data = str(value) + " " + str(0) + "\n"
    output_data += " ".join(map(str, taken))
    return output_data


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fileLocation = sys.argv[1].strip()
        inputDataFile = open(fileLocation, "r")
        inputData = "".join(inputDataFile.readlines())
        inputDataFile.close()
        print(solve_it(inputData))
    else:
        print(
            "This test requires an input file.  Please select one from  data_ninjas (i.e. python solver.py ./data/ninjas_1_4)"
        )
        # EXAMPLE of execution from terminal:
        #      python solver.py ./data/ninja_1_4
