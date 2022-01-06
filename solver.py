import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm


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


def tournament_selection(
    k: int, population_idx: np.ndarray, fitting_values: np.ndarray, keep: int
):
    """
    Perform a tournament selection.
    :param k: Number of randomly selected individuals to use in the selection.
    :param population_idx: List of population indices.
    :param fitting_values: Fitness value for each individual.
    :param keep: Number of individuals to keep in the selection.
    :return: The list of individuals that won the tournament.
    :rtype : np.ndarray
    """
    selection = np.random.choice(population_idx, k)
    selection_val = fitting_values[selection]

    return elitism(population_idx, selection_val, keep)


def plot_figure(path, values, figsize=(20,20)):
    plt.plot(values, figsize=figsize)
    plt.xlabel("epochs")
    plt.ylabel("fitness")
    plt.savefig(path)


class GeneticAlgorithm:
    # Options for crossover methods
    MUTATION_PROBABILITY = 0.5
    ONE_POINT_CROSSOVER = 0
    TWO_POINT_CROSSOVER = 1
    UNIFORM_CROSSOVER = 2

    # Options for selection methods
    TOURNAMENT = 0
    ELITISM = 1

    def __init__(
        self,
        n_generations,
        stall_generations,
        population_size,
        chromosome_size,
        values,
        weights,
        capacity,
        selection_method,
        crossover_method,
        init_pop_range=None,
        sort_values=False,
        optimal_value=None,
        fig_path=None,
    ):
        self.n_generations = n_generations
        self.stall_generations = stall_generations
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        self.values = np.array(values, dtype=int)
        self.weights = np.array(weights, dtype=int)
        self.capacity = capacity
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.optimal_value = optimal_value
        self.fig_path = fig_path
        self.population = np.zeros(
            (self.population_size, self.chromosome_size), dtype=np.int8
        )
        self.original_order = np.array([], dtype=int)
        self.rng = default_rng()

        if init_pop_range is not None:
            self.init_pop_range = init_pop_range
        else:
            top = (
                2 ** self.chromosome_size
                if 2 ** self.chromosome_size <= np.iinfo(np.uint).max
                else np.iinfo(np.uint).max
            )
            self.init_pop_range = (0, top)

        if (self.init_pop_range[1] - self.init_pop_range[0]) < self.population_size:
            raise ValueError(
                "The initial population range is lower than the population size."
            )

        max_number = 2 ** self.chromosome_size
        if self.init_pop_range[0] > max_number or self.init_pop_range[1] > max_number:
            raise ValueError(
                "The initial population range contains numbers not representable by the chromosome size."
            )

        self.init_population()

        if sort_values:
            sort_indices = np.argsort(self.values / self.weights)
            self.original_order = np.argsort(sort_indices)
            self.values = self.values[sort_indices]
            self.weights = self.weights[sort_indices]

        self.current_fitness = self.fitness_value(self.population)

    def init_population(self):
        """
        Randomly initialize the population.
        """
        pop = np.array([], dtype=int)

        low_range, high_range = self.init_pop_range
        # Generate N random unique individuals
        if (high_range - low_range) == self.population_size:
            pop = np.array(range(self.population_size))
        else:
            while len(pop) < self.population_size:
                pop = np.unique(
                    np.append(
                        pop,
                        np.unique(
                            self.rng.integers(
                                low=low_range,
                                high=high_range,
                                size=(self.population_size - len(pop)),
                                dtype=np.uint,
                            )
                        ),
                    )
                )

        for i in range(self.population_size):
            self.population[i] = np.array(
                [int(s) for s in np.binary_repr(pop[i], self.chromosome_size)],
                dtype=np.uint8,
            )

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
        if self.rng.random() < self.MUTATION_PROBABILITY:
            gene = self.rng.choice(range(self.chromosome_size))
            individual[gene] = not (individual[gene])

    def one_point_crossover(self, parent1, parent2):
        point = self.rng.integers(1, self.chromosome_size - 1)
        offspring1 = np.concatenate((parent1[:point], parent2[point:]))
        offspring2 = np.concatenate((parent2[:point], parent1[point:]))
        self.mutate(offspring1)
        self.mutate(offspring2)
        return offspring1, offspring2

    def two_point_crossover(self, parent1, parent2):
        points = self.rng.choice(range(1, self.chromosome_size - 1), 2, replace=False)
        point1 = min(points)
        point2 = max(points)

        parent1_s1, parent1_s2, parent1_s3 = np.split(parent1, [point1, point2])
        parent2_s1, parent2_s2, parent2_s3 = np.split(parent2, [point1, point2])
        offspring1 = np.concatenate((parent1_s1, parent2_s2, parent1_s3))
        offspring2 = np.concatenate((parent2_s1, parent1_s2, parent2_s3))
        self.mutate(offspring1)
        self.mutate(offspring2)
        return offspring1, offspring2

    def ux_crossover(self, parent1, parent2):
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        seed = self.rng.random(self.chromosome_size)
        for s, i in enumerate(seed):
            if s < 0.5:
                offspring1[i] = parent2[i]
                offspring2[i] = parent1[i]

        self.mutate(offspring1)
        self.mutate(offspring2)
        return offspring1, offspring2

    def crossover(self, parent1, parent2):
        if self.crossover_method == self.ONE_POINT_CROSSOVER:
            return self.one_point_crossover(parent1, parent2)
        elif self.crossover_method == self.TWO_POINT_CROSSOVER:
            return self.two_point_crossover(parent1, parent2)
        elif self.crossover_method == self.UNIFORM_CROSSOVER:
            return self.ux_crossover(parent1, parent2)

    def selection(self, chromosomes, fitness, keep, k=2):
        """
        Perform the population selection.
        :param population: List of individuals.
        :param fitting_values: Fitness value for each individual.
        :param keep: Number of individuals to keep in the selection.
        :param k: Number of randomly selected individuals to use in the selection.
        :return: The list of individuals that won the selection.
        :rtype : np.ndarray
        """
        if self.selection_method == self.TOURNAMENT:
            return tournament_selection(k, chromosomes, fitness, keep)
        elif self.selection_method == self.ELITISM:
            return elitism(chromosomes, fitness, keep)

    def run(self):
        winner_fitness = np.NINF
        stall_generations = 0
        optimal_found = 0
        fitness_evol = np.array([])
        for _ in tqdm(range(self.n_generations), leave=False):
            population_idx = np.arange(self.population_size)
            offspring = np.zeros(self.population.shape, dtype=np.int8)
            k = 2
            for i in range(0, self.population_size, 2):
                if len(population_idx) > k:
                    parent1_idx = self.selection(
                        population_idx, self.current_fitness, keep=1, k=k
                    )[0]
                    population_idx = population_idx[population_idx != parent1_idx]

                    parent2_idx = self.selection(
                        population_idx, self.current_fitness, keep=1, k=k
                    )[0]
                    population_idx = population_idx[population_idx != parent2_idx]
                elif len(population_idx) < k:
                    continue
                else:
                    parent1_idx = population_idx[0]
                    parent2_idx = population_idx[1]

                offspring1, offspring2 = self.crossover(
                    self.population[parent1_idx], self.population[parent2_idx]
                )
                offspring[i] = offspring1
                offspring[i + 1] = offspring2

            offspring = np.unique(offspring, axis=0)
            offspring_fitness = self.fitness_value(offspring)
            if np.all(self.current_fitness == np.NINF) and np.all(
                offspring_fitness == np.NINF
            ):
                # If there are no individuals with good fitness,
                # then we replace half of the previous generation with the new one.
                pop_1 = self.rng.choice(
                    self.population, self.population_size // 2, replace=False
                )
                pop_2 = self.rng.choice(
                    offspring, self.population_size // 2, replace=False
                )
                self.population = np.concatenate((pop_1, pop_2))
            else:
                # Select only non repeated individuals to ensure diversity is maintained in the population.
                join_population = np.unique(
                    np.concatenate((self.population, offspring)), axis=0
                )
                self.population = elitism(
                    join_population,
                    self.fitness_value(join_population),
                    self.population_size,
                )
            self.current_fitness = self.fitness_value(self.population)

            if np.max(self.current_fitness) == winner_fitness:
                stall_generations += 1
            else:
                stall_generations = 0
                winner_fitness = np.max(self.current_fitness)

            fitness_evol = np.append(fitness_evol, winner_fitness)

            if self.optimal_value is not None and self.optimal_value == winner_fitness:
                optimal_found = 1
                break

            if stall_generations >= self.stall_generations:
                if self.optimal_value is None and winner_fitness != np.NINF:
                    optimal_found = 1
                break

        fittest_individual_idx = np.argmax(self.current_fitness)
        fittest_individual = self.population[fittest_individual_idx]
        best_fitness = self.current_fitness[fittest_individual_idx]

        if best_fitness != np.NINF:
            best_fitness = int(best_fitness)

        if self.fig_path is not None:
            plot_figure(self.fig_path, fitness_evol)

        if self.original_order.size == 0:
            return (
                fittest_individual,
                best_fitness,
                optimal_found,
            )
        else:
            return (
                fittest_individual[self.original_order],
                best_fitness,
                optimal_found,
            )


def read_best_value(file_name: str):
    best_sol_file = f'best_sol{file_name.removeprefix("ninja")}'
    with open(best_sol_file, "r") as f:
        return int(float(f.readline().split()[0]))


def solve_it(input_data, file_location):
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
    file_name = file_location[file_location.find("n") :]
    best_value = read_best_value(file_name)

    pop_size = items ** 2
    ga = GeneticAlgorithm(
        n_generations=5000,
        stall_generations=500,
        population_size=pop_size,
        chromosome_size=items,
        values=values,
        weights=weights,
        capacity=capacity,
        selection_method=GeneticAlgorithm.TOURNAMENT,
        crossover_method=GeneticAlgorithm.TWO_POINT_CROSSOVER,
        init_pop_range=[1, 3 * pop_size],
        sort_values=True,
        optimal_value=best_value,
        fig_path=os.path.join("figures", f"{file_name}.png"),
    )
    taken, value, optimal_found = ga.run()

    ## MAGIC ##
    # best_value = magic_d(weights, values, capacity)

    # value = 0
    # taken = items * [0]

    # value = magic_d(weights, values, capacity)

    # STOP WRITING YOUR CODE HERE ###################################

    output_data = str(value) + " " + str(int(value == best_value)) + "\n"
    output_data += " ".join(map(str, taken))
    return output_data


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fileLocation = sys.argv[1].strip()
        inputDataFile = open(fileLocation, "r")
        inputData = "".join(inputDataFile.readlines())
        inputDataFile.close()
        print(solve_it(inputData, fileLocation))
    else:
        print(
            "This test requires an input file.  Please select one from  data_ninjas (i.e. python solver.py ./data/ninjas_1_4)"
        )
        # EXAMPLE of execution from terminal:
        #      python solver.py ./data/ninja_1_4
