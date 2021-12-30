import random
import sys
from random import getrandbits

import numpy as np
from numpy.random import default_rng


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


def elitism(population: np.ndarray, fitting_values: np.ndarray, keep: int):
    """
    Perform elitism selection on a population.
    :param numpy.array population: List of individuals.
    :param fitting_values: Fitness value for each individual.
    :param keep: Number of individuals to keep.
    :return: The best N individuals defined by the ``keep`` parameter.
    :rtype : np.ndarray
    """
    return population[np.argpartition(fitting_values, -keep)[-keep:]]


def tournament_selection(k: int, population: np.ndarray, fitting_values: np.ndarray, keep: int):
    """
    Perform a tournament selection.
    :param k: Number of randomly selected individuals to use in the selection.
    :param population: List of individuals.
    :param fitting_values: Fitness value for each individual.
    :param keep: Number of individuals to keep in the selection.
    :return: The list of individuals that won the tournament.
    :rtype : np.ndarray
    """
    selection = np.random.choice(range(population.shape[0]), k)
    selection_val = fitting_values[selection]

    return elitism(population, selection_val, keep)


class GeneticAlgorithm:
    # Options for crossover methods
    N_INDIVIDUALS = 50
    MUTATION_PROBABILITY = 0.2
    ONE_POINT_CROSSOVER = 0
    TWO_POINT_CROSSOVER = 1
    UNIFORM_CROSSOVER = 2

    # Options for selection methods
    TOURNAMENT = 0
    ELITISM = 1

    def __init__(self, n, m, values, weights, capacity):
        self.n = n
        self.m = m
        self.values = values
        self.weights = weights
        self.capacity = capacity
        self.population = np.zeros((self.n, self.m))

        self.init_population()
        self.current_fitness = self.fitness_value(self.population)

    def init_population(self):
        """
        Randomly initialize the population.
        """
        pop = np.array([], dtype=int)
        rng = default_rng()

        # Generate N random unique individuals
        while len(pop) < self.n:
            pop = np.unique(
                np.append(pop, np.unique(rng.integers(low=0, high=(2 ** self.m - 1), size=(self.n - len(pop))))))

        for i in range(self.n):
            self.population[i] = np.array([int(s) for s in np.binary_repr(pop[i], self.m)], dtype=np.uint8)

    def fitness_value(self, chromosomes):
        fitness_values = np.zeros(chromosomes.shape[0])
        for i, chromosome in enumerate(chromosomes):
            # If the chromosome exceeds the capacity its value is 0.
            # Otherwise, the value is the total value of the items it selected.
            if np.dot(self.weights, chromosome) > self.capacity:
                fitness_values[i] = np.NINF
            else:
                fitness_values[i] = np.dot(self.values, chromosome)

        return fitness_values

    def mutate(self, individual):
        for i in range(individual.shape[0]):
            if random.random() <= self.MUTATION_PROBABILITY:
                individual[i] = 1 ^ individual[i]

    def one_point_crossover(self, parent1, parent2):
        point = np.random.randint(1, parent1.shape[0] - 1)
        offspring1 = np.concatenate((parent1[:point], parent2[point:]))
        offspring2 = np.concatenate((parent2[:point], parent1[point:]))
        return self.mutate(offspring1), self.mutate(offspring2)

    def two_point_crossover(self, parent1, parent2):
        point1 = np.random.randint(1, parent1.shape[0] - 1)

        while True:
            point2 = np.random.randint(1, parent1.shape[0] - 1)
            if point2 != point1:
                break

        parent1_s1, parent1_s2, parent1_s3 = np.split(parent1, np.sort([point1, point2]))
        parent2_s1, parent2_s2, parent2_s3 = np.split(parent2, np.sort([point1, point2]))
        offspring1 = np.concatenate((parent1_s1, parent2_s2, parent1_s3))
        offspring2 = np.concatenate((parent2_s1, parent1_s2, parent2_s3))
        return self.mutate(offspring1), self.mutate(offspring2)

    def ux_crossover(self, parent1, parent2):
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        for i in range(parent1.shape[0]):
            if random.random() < 0.5:
                offspring1[i] = parent2[i]
                offspring2[i] = parent1[i]

        return self.mutate(offspring1), self.mutate(offspring2)

    def crossover(self, parent1, parent2, method=0):
        if method == self.ONE_POINT_CROSSOVER:
            return self.one_point_crossover(parent1, parent2)
        elif method == self.TWO_POINT_CROSSOVER:
            return self.two_point_crossover(parent1, parent2)
        elif method == self.UNIFORM_CROSSOVER:
            return self.ux_crossover(parent1, parent2)

    def selection(self, chromosomes, fitness, k, keep, method=0):
        if method == self.TOURNAMENT:
            return tournament_selection(k, chromosomes, fitness, keep)
        elif method == self.ELITISM:
            return elitism(chromosomes, fitness, keep)

    def mate(self, chromosomes, crossover=0):
        offspring = np.zeros(self.population.shape)
        for i in range(0, self.N_INDIVIDUALS, 2):
            parent1 = tournament_selection(2, self.population, self.current_fitness, 1)
            parent2 = tournament_selection(2, self.population, self.current_fitness, 1)
            offspring1, offspring2 = self.crossover(parent1, parent2, crossover)
            offspring[i] = offspring1
            offspring[i + 1] = offspring2

        offspring_fitness = self.fitness_value(offspring)
        self.population = elitism(np.concatenate((chromosomes, offspring)),
                                  np.concatenate((self.current_fitness, offspring_fitness)), self.N_INDIVIDUALS)
        self.current_fitness = self.fitness_value(self.population)


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

    # population = init_population(N_INDIVUDUALS, items)
    # population = mate(population, items_values, items_wieghts, capacity)

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
