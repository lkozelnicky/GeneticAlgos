"""Python library for creating genetic algorithms.

https://github.com/lkozelnicky/GeneticAlgos

GeneticAlgos is a simple and powerful Python library for creating
genetic algorithms to solve complex optimization problems. GeneticAlgos
is built on NumPy and it is under active development.

Features:
    - Uses smart defaults for genetic algorithms parameters
    - A simple-to-use API to modify genetic algorithms parameters.
    - Lightweight and just one dependency - Numpy.
    - Excellent test coverage.
    - Tested on Python 3.7, 3.8 and 3.9
"""
from random import sample
from typing import Callable, List, Tuple

import numpy as np

from geneticalgos import exceptions


class GeneticAlgo:
    """A user-created :class:`GeneticAlgo <GeneticAlgo>` object.

    Object instantiated by the :class:`GeneticAlgo <GeneticAlgo>`
    simulate genetic algorithm evolution.

    :param callable fitness_function: user defined objective function
        which calculate suitability of the given chromosome.
    :param numpy.ndarray gene_intervals: 2-dimensional ``numpy.ndarray``
        specifies chromosome length and value range for each gene.
    :param str objective_goal: optional - whether maximum (``maximize``)
        or minimum (``minimize``) is considered best value for fitness
        function. Supported values are ``minimize``, ``maximize``.
        The default value is ``maximize``.
    :param str chromosome_type: optional - chromosome encoding.
        Supported values are ``int``, ``float``. The default value
        is ``float``.
    :param int population_size: optional - population size.
        The default value is ``100``.

    :var selection_k_max: max tournament size for tournament selection
        method. Hard coded value is ``20``.
    """

    selection_k_max: int = 20

    def __init__(
        self,
        fitness_function: Callable[[np.ndarray], float],
        gene_intervals: np.ndarray,
        objective_goal: str = "maximize",
        chromosome_type: str = "float",
        population_size: int = 100,
    ) -> None:
        """Initialize genetic algorithms model.

        :raises FitnessFunctionNotCallable: if fitness_function
            is not callable
        :raises TypeError: if any of the parameters have wrong type
        :raises InvalidGeneIntervalsShape: if gene_intervals
            numpy.ndarray has wrong shape
        :raises InvalidObjectiveGoal: if wrong objective_goal value
        :raises InvalidChromosomeType: if wrong chromosome_type value
        :raises ValueError: if wrong population_size
        """
        if not callable(fitness_function):
            raise exceptions.FitnessFunctionNotCallable(
                f'fitness_function "{fitness_function}" is not callable'
            )

        self._fitness_function = fitness_function

        if not isinstance(gene_intervals, np.ndarray):
            raise TypeError(
                f"gene_intervals must be numpy.ndarray, but got {type(gene_intervals).__name__}"
            )

        if len(gene_intervals.shape) != 2 or gene_intervals.shape[1] != 2:
            raise exceptions.InvalidGeneIntervalsShape(
                f"not valid gene_intervals shape. Expected 2-dimensional array with axis 1 equal "
                f"to two, but got {gene_intervals.shape}"
            )

        self._gene_intervals = gene_intervals

        if objective_goal not in ["minimize", "maximize"]:
            raise exceptions.InvalidObjectiveGoal(
                f'not valid objective_goal. Expected "minimize" or "maximize", '
                f'but got "{objective_goal}"'
            )

        self._objective_goal = objective_goal

        if not isinstance(chromosome_type, str):
            raise TypeError(
                f"chromosome_type must be str, but got {type(chromosome_type).__name__}"
            )

        if chromosome_type not in ["int", "float"]:
            raise exceptions.InvalidChromosomeType(
                'not valid chromosome_type. Expected "int" or "float", '
                f'but got "{chromosome_type}"'
            )

        self._chromosome_type = chromosome_type
        if self._chromosome_type == "int":
            self._rand_function = np.random.randint
        else:
            self._rand_function = np.random.uniform  # type: ignore

        if not isinstance(population_size, int):
            raise TypeError(
                f"population_size type must be int, but got {type(population_size).__name__}"
            )

        if population_size < 4:
            raise ValueError(
                f"population_size must be greater than 3, but got {population_size}"
            )

        self._population_size = population_size

        self._fitness_values = np.zeros(self._population_size)

        # Default parameters for :meth:`simulate() <simulate>`
        self.selection_type = "random"
        self._selection_k = 3
        self.crossover_type = "random"
        self._crossover_prob = 0.9
        self._mutation_prob = 0.2
        self._new_pop_type = "random"
        self._n_pairs = divmod(self._population_size, 3)[0]
        self._n_elite = 2

        self.best_chromosomes_history: List[float] = list()

    @property
    def selection_type(self) -> str:
        """Selection method.

        Specifies which selection method is going to be used in
        :meth:`simulate() <simulate>`. Supported values are "random",
        "roulette_fitness", "roulette_rank" and "tournament".
        The default value is "random".

        :getter: Returns current selection method.
        :setter: Sets new selection method.
        :raises TypeError: if new value has incorrect type
        :raises ValueError: if new value is not supported selection
            method
        """
        return self._selection_type

    @selection_type.setter
    def selection_type(self, new_value: str) -> None:
        """Set selection method."""
        if not isinstance(new_value, str):
            raise TypeError(
                f"selection_type must be str, but got {type(new_value).__name__}"
            )

        try:
            self._selection = {
                "random": self._random_selection,
                "roulette_fitness": self._roulette_fitness_selection,
                "roulette_rank": self._roulette_rank_selection,
                "tournament": self._tournament_selection,
            }[new_value]
            self._selection_type = new_value
        except KeyError:
            raise ValueError(
                f'supported selection types "roulette_fitness","roulette_rank", "tournament", '
                f'"random", but got "{new_value}"'
            )

    @property
    def selection_k(self) -> int:
        """Tournaments size for tournament selection method.

        Maximum tournament size is hard coded in :attr:`selection_k_max`
        and it is 20.

        :getter: Returns current tournament size
        :setter: Sets new tournament size
        :raises TypeError: if new value has incorrect type
        :raises ValueError: if new value is not within allowed range
        """
        return self._selection_k

    @selection_k.setter
    def selection_k(self, new_value) -> None:
        """Set tournament size for tournament selection method."""
        if not isinstance(new_value, int):
            raise TypeError(
                f"selection_k size must be int, but got {type(new_value).__name__}"
            )

        if self._selection_type != "tournament":
            raise ValueError(
                f'selection_k size is supported for "tournament" selection_type, '
                f'but now selection_type is "{self._selection_type}"'
            )

        if 0 <= new_value < self.selection_k_max:
            self._selection_k = new_value
            return

        raise ValueError(
            f"selection_k size must be greater than zero and smaller than "
            f"{self.selection_k_max}, but got {new_value}"
        )

    @property
    def n_pairs(self) -> int:
        """Number of parent pairs selected for reproduction.

        specifies the number of pairs selected to reproduce (crossover,
        mutation). If 5 it means that 5 * 2 = 10 chromosomes are going
        to be selected for reproduction. If 0 there is no chromosome
        selected for reproduction. Upper limit is:
        (:attr:`population_size` - :attr:`n_elite`) / 2.
        The default value is :attr:`population_size` / 3.

        :getter: Returns current number of pairs for selection
        :setter: Sets new number of pairs for selection
        :raises TypeError: if new value has incorrect type
        :raises ValueError: if new value is not within allowed range
        """
        return self._n_pairs

    @n_pairs.setter
    def n_pairs(self, new_value: int) -> None:
        """Set number of pairs for selection."""
        if not isinstance(new_value, int):
            raise TypeError(f"n_pairs must be int, but got {type(new_value).__name__}")

        max_n_pairs = int((self._population_size - self._n_elite) / 2)

        if 0 < new_value <= max_n_pairs:
            self._n_pairs = new_value
            return

        raise ValueError(
            f"n_pairs must be int from interval (0, {max_n_pairs}>, "
            f"but got {new_value}"
        )

    @property
    def crossover_type(self) -> str:
        """Crossover method.

        Specifies which crossover method is going to be used
        in :meth:`simulate() <simulate>`. Supported values are "random",
        "one_point", "two_points", and "uniform". The default value
        is "random".

        :getter: Returns current crossover method
        :setter: Sets new crossover method
        :raises TypeError: if new value has incorrect type
        :raises ValueError: if new value is not supported crossover
            method
        """
        return self._crossover_type

    @crossover_type.setter
    def crossover_type(self, new_value: str) -> None:
        """Set crossover method."""
        if not isinstance(new_value, str):
            raise TypeError(
                f"crossover_type must be str, but got {type(new_value).__name__}"
            )

        try:
            self._crossover = {
                "random": self._random_crossover,
                "one_point": self._one_point_crossover,
                "two_points": self._two_points_crossover,
                "uniform": self._uniform_crossover,
            }[new_value]
            self._crossover_type = new_value
        except KeyError:
            raise ValueError(
                f'supported crossover types "one_point","two_points", "uniform", "random", '
                f'but got "{new_value}"'
            )

    @property
    def crossover_prob(self) -> float:
        """crossover probability.

        Crossover probability specifies how likely is that crossover
        is applied to selected parents. Must be float between <0-1>.
        If 0 then crossover is never applied. If 1 then crossover is
        applied to every selected parent pair. The default value is 0.9,
        meaning there is 90% chance that crossover is applied
        to the given parent pair.

        :getter: Returns current crossover probability
        :setter: Sets new crossover probability
        :raises TypeError: if new value has incorrect type
        :raises ValueError: if new value is not within allowed
            range <0-1>
        """
        return self._crossover_prob

    @crossover_prob.setter
    def crossover_prob(self, new_value: float) -> None:
        """Set crossover probability."""
        if not isinstance(new_value, float):
            raise TypeError(
                f"crossover_prob must be float, but got {type(new_value).__name__}"
            )

        if 0 <= new_value <= 1:
            self._crossover_prob = new_value
            return

        raise ValueError(
            f"crossover_prob must be from <0-1> interval, but got {new_value}"
        )

    @property
    def mutation_prob(self) -> float:
        """Mutation probability for particular gene.

        Mutation is applied to each gene with mutation probability.
        Must be float between <0-1>. If 0 then mutation is not applied
        at all. If 1 then mutation is applied to every gene
        in chromosome. The default value is 0.2, meaning  meaning
        there is 20% chance that gene is going to mutate.

        :getter: Returns current mutation probability
        :setter: Sets new mutation probability
        :raises TypeError: if new value has incorrect type
        :raises ValueError: if new value is not within allowed
            range <0-1>
        """
        return self._mutation_prob

    @mutation_prob.setter
    def mutation_prob(self, new_value: float) -> None:
        """Set mutation probability."""
        if not isinstance(new_value, float):
            raise TypeError(
                f"mutation_prob must be float, but got {type(new_value).__name__}"
            )

        if 0 <= new_value <= 1:
            self._mutation_prob = new_value
            return

        raise ValueError(
            f"mutation_prob must be from <0-1> interval, but got {new_value}"
        )

    @property
    def new_pop_type(self) -> str:
        """Method for creating new population.

        Creates new population when all parents finish their
        reproduction process (crossover, mutation). Supported methods
        are "random", "always_offsprings" and "tournament". The default
        value is "random".

        :getter: Returns current method for creating new population
        :setter: Sets new method for creating new population
        :raises TypeError: if new value has incorrect type
        :raises ValueError: if new value is not supported method
            for creating new population
        """
        return self._new_pop_type

    @new_pop_type.setter
    def new_pop_type(self, new_value: str) -> None:
        """Set method for creating new population."""
        if not isinstance(new_value, str):
            raise TypeError(
                f"new_pop_type must be str, but got {type(new_value).__name__}"
            )

        if new_value in ["random", "always_offsprings", "tournament"]:
            self._new_pop_type = new_value
            return

        raise ValueError(
            'supported new_pop_type "always_offsprings" or "tournament", '
            f'but got "{new_value}"'
        )

    @property
    def n_elite(self) -> int:
        """Configure elitism.

        :attr:`n_elite` - specifies how many chromosome are carried over
        to the next generation without any gene change. This method
        guarantees that the solution quality obtained by the genetic
        algorithms will not decrease from one generation to the next.

        If 0 there is no elitism.
        Upper limit is: :attr:`population_size` - (:attr:`n_pairs` * 2).
        The default value is 2.

        :getter: Returns current elite count
        :setter: Sets new elite count
        :raises TypeError: if new value has incorrect type
        :raises ValueError: if new value is not within allowed range
        """
        return self._n_elite

    @n_elite.setter
    def n_elite(self, new_value: int) -> None:
        """Set elite count."""
        if not isinstance(new_value, int):
            raise TypeError(f"n_elite must be int, but got {type(new_value).__name__}")

        max_n_elite = int(self._population_size - self._n_pairs * 2)

        if 0 <= new_value <= max_n_elite:
            self._n_elite = new_value
            return

        raise ValueError(
            f"n_elite must be int from interval (0, {max_n_elite}>, "
            f"but got {new_value}"
        )

    def simulate(self, n_iterations: int = 100) -> None:
        """Simulate genetic algorithms evolution.

        Workflow for simulating genetic evolution is - select
        parents, crossover, mutation and then create new population.
        Repeating this loop :data:`n_iterations <n_iterations>` times.
        Every time is method executed it starts genetic evolution
        from beginning.

        :param int n_iterations: number of generation cycles
        :raises TypeError: if n_iterations is not int
        :raises InvalidNumberOfIterations: if n_iterations
            is not positive int value
        """

        if not isinstance(n_iterations, int):
            raise TypeError(
                f"n_iterations type must be int, but got {type(n_iterations).__name__}"
            )

        if n_iterations < 1:
            raise exceptions.InvalidNumberOfIterations(
                f"n_iterations must be positive int, but got {n_iterations}"
            )

        self._create_initial_population()

        self.best_chromosomes_history = list()

        generation_number = 0

        while generation_number < n_iterations:
            # initialize new population
            self._new_population = np.zeros(
                shape=[self._population_size, len(self._gene_intervals)]
            )

            self._evaluate_population()

            if self._n_elite:
                self._add_elite()

            all_parents = self._selection()

            for i in range(0, len(all_parents), 2):
                parent_1_ix = all_parents[i]
                parent_2_ix = all_parents[i + 1]
                parent_1 = self._population[parent_1_ix]
                parent_2 = self._population[parent_2_ix]

                offspring_1, offspring_2 = self._crossover(parent_1, parent_2)

                offspring_1 = self._mutation(offspring_1)
                offspring_2 = self._mutation(offspring_2)

                if (
                    self.new_pop_type == "random" and np.random.uniform() < 0.5
                ) or self.new_pop_type == "tournament":
                    (
                        self._new_population[i],
                        self._new_population[i + 1],
                    ) = self._parents_offspring_tournament(
                        parents_ix=[parent_1_ix, parent_2_ix],
                        offsprings=[offspring_1, offspring_2],
                    )
                # else:
                #  (new_pop_type == "random" and random.uniform() > 0.5)
                #  or "always_offspring"
                else:
                    self._new_population[i] = offspring_1
                    self._new_population[i + 1] = offspring_2

            self._create_new_population()

            generation_number += 1

        self._get_final_results()

    def _create_initial_population(self) -> None:
        """Create initial population with random values."""
        lower_bounds = [x[0] for x in self._gene_intervals]
        higher_bounds = [x[1] for x in self._gene_intervals]

        self._population = self._rand_function(
            low=lower_bounds,
            high=higher_bounds,
            size=(self._population_size, len(self._gene_intervals)),
        )

    def _evaluate_population(self) -> None:
        """Calculate fitness values for all chromosomes.

        Fitness values for each chromosome are determined by the user
        defined fitness_function in :meth:`__init__() <__init__>`

        :raises FitnessFunctionReturnsNone: if user fitness
            function returns None
        """
        self._fitness_values = np.array(
            [self._fitness_function(x) for x in self._population]
        )

        if np.any(np.equal(self._fitness_values, None)):  # type: ignore
            raise exceptions.FitnessFunctionReturnsNone("fitness_function returns None")

        if self._objective_goal == "maximize":
            best_chromosome = np.amax(self._fitness_values)
        else:
            best_chromosome = np.amin(self._fitness_values)

        self.best_chromosomes_history.append(float(best_chromosome))

    def _add_elite(self) -> None:
        """Applying elitism.

        Instantaneously move :attr:`n_elite` best chromosomes to next
        generation.
        """
        if self._objective_goal == "maximize":
            elite_ix = np.argsort(self._fitness_values)[-self._n_elite :]
        else:
            elite_ix = np.argsort(self._fitness_values)[: self._n_elite]

        for i, elite_index in enumerate(elite_ix):
            self._new_population[self._n_pairs * 2 + i] = self._population[elite_index]

    def _random_selection(self) -> List[int]:
        """Random selection method.

        Randomly choose selection method used in
        :meth:`simulate() <simulate>` for selecting parents
        to reproduction (crossover, mutation). This is repeating
        every generation cycle, so each cycle might have different
        selection method.

        :return: all parents indexes in population for reproduction
        """
        return [
            self._roulette_fitness_selection,
            self._roulette_rank_selection,
            self._tournament_selection,
        ][np.random.randint(3)]()

    def _roulette_fitness_selection(self) -> List[int]:
        """Roulette wheel fitness proportionate selection method.

        Creates "roulette wheel" where each chromosome has
        assigned proportion of the wheel based on his fitness value.
        Randomly select :attr:`n_pairs` * 2 parents.

        :return: all parents indexes in population for reproduction
        """
        if self._objective_goal == "maximize":
            fitness_proportional = self._fitness_values / np.sum(self._fitness_values)
        else:
            inverted_fitness = 1 - self._fitness_values / np.sum(self._fitness_values)
            fitness_proportional = inverted_fitness / np.sum(inverted_fitness)

        roulette = np.cumsum(fitness_proportional)

        return [
            np.where(roulette > np.random.uniform())[0][0]
            for _ in range(self._n_pairs * 2)
        ]

    def _roulette_rank_selection(self) -> List[int]:
        """Roulette wheel rank selection selection.

        First creates sorted array of indexes based of fitness values.
        Then creates "roulette wheel" where each chromosome has
        assigned proportion of the wheel based on the rank
        in population. Selected chromosome is identified based
        on sorted rank array. Randomly select :attr:`<n_pairs>` * 2
        parents.

        :return: all parents indexes in population for reproduction
        """
        if self._objective_goal == "maximize":
            sorted_fitness = np.argsort(self._fitness_values)
        else:
            # sort in reverse order
            sorted_fitness = np.argsort(-1 * self._fitness_values)

        rank_proportional = np.array(
            [
                (2 * i) / (self._population_size * (self._population_size + 1))
                for i in range(1, self._population_size + 1, 1)
            ]
        )

        roulette = np.cumsum(rank_proportional)

        return [
            sorted_fitness[np.where(roulette > np.random.uniform())[0][0]]
            for _ in range(self._n_pairs * 2)
        ]

    def _tournament_selection(self) -> List[int]:
        """Tournament selection method.

        Randomly choose :attr:`selection_k` chromosomes and parent
        is the one with the best fitness value amongst them.
        Tournament repeat :attr:`<n_pairs>` * 2 times and return all
        winning chromosome.

        :return: all parents indexes in population for reproduction
        """

        def tournament():
            winner_ix = np.random.randint(self._population_size)
            for ix in np.random.randint(
                0, self._population_size, self._selection_k - 1
            ):
                if self._objective_goal == "maximize":
                    if self._fitness_values[ix] > self._fitness_values[winner_ix]:
                        winner_ix = ix
                else:
                    if self._fitness_values[ix] < self._fitness_values[winner_ix]:
                        winner_ix = ix
            return winner_ix

        return [tournament() for _ in range(self._n_pairs * 2)]

    def _random_crossover(
        self, parent_1: np.ndarray, parent_2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Randomly choose crossover method.

        Randomly choose crossover method which is going to used
        in :meth:`simulate() <simulate>` in particular generation
        cycle. This is repeating every generation cycle, so each
        cycle might be different crossover method.

        :param parent_1: 1st chromosome for crossover
        :param parent_2: 2nd chromosome for crossover
        :return: two crossed over chromosomes (offsprings)
        """
        if np.random.uniform() < self._crossover_prob:
            return [
                self._one_point_crossover,
                self._two_points_crossover,
                self._uniform_crossover,
            ][np.random.randint(3)](parent_1, parent_2)

        return parent_1, parent_2

    def _one_point_crossover(
        self, parent_1: np.ndarray, parent_2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One-point crossover method.

        Randomly picks one crossover point in chromosomes. Genes
        to the right of that point are swapped between the two parent
        chromosomes. This results in two offspring, each carrying some
        genetic information from both parents.

        :param parent_1: 1st chromosome for crossover
        :param parent_2: 2nd chromosome for crossover
        :return: two crossed over chromosomes (offsprings)
        """
        if np.random.uniform() < self._crossover_prob:
            x_point = np.random.randint(0, len(parent_1) + 1)
            offspring_1 = np.concatenate([parent_1[:x_point], parent_2[x_point:]])

            offspring_2 = np.concatenate([parent_2[:x_point], parent_1[x_point:]])

            return offspring_1, offspring_2

        return parent_1, parent_2

    def _two_points_crossover(
        self, parent_1: np.ndarray, parent_2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Two-points crossover method.

        Randomly picks two crossover points. The genes in-between
        the two points are swapped between the parent organisms.

        :param parent_1: 1st chromosome for crossover
        :param parent_2: 2nd chromosome for crossover
        :return: two crossed over chromosomes (offsprings)
        """
        if np.random.uniform() < self._crossover_prob:
            x_point_1, x_point_2 = sorted(sample(range(0, len(parent_1) + 1), 2))

            offspring_1 = np.concatenate(
                [
                    parent_1[:x_point_1],
                    parent_2[x_point_1:x_point_2],
                    parent_1[x_point_2:],
                ]
            )

            offspring_2 = np.concatenate(
                [
                    parent_2[:x_point_1],
                    parent_1[x_point_1:x_point_2],
                    parent_2[x_point_2:],
                ]
            )
            return offspring_1, offspring_2

        return parent_1, parent_2

    def _uniform_crossover(
        self, parent_1: np.ndarray, parent_2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover method.

        Uniform crossover treats each gene separately. Both parents
        have a 50% chance of their gene ending up in their offspring.

        :param parent_1: 1st chromosome for crossover
        :param parent_2: 2nd chromosome for crossover
        :return: two crossed over chromosomes (offsprings)
        """
        if np.random.uniform() < self._crossover_prob:
            offspring_1 = parent_1.copy()
            offspring_2 = parent_2.copy()

            for i in range(len(parent_1)):
                if np.random.uniform() > 0.5:
                    offspring_1[i] = parent_2[i].copy()
                    offspring_2[i] = parent_1[i].copy()

            return offspring_1, offspring_2

        return parent_1, parent_2

    def _mutation(self, chromosome: np.ndarray) -> np.ndarray:
        """Mutation genetic operator.

        Mutation treats each gene separately. Gene is going
        to change (mutate) to random value within allowed range
        with :attr:`mutation_prob`

        :param chromosome: chromosome for mutation
        :return: chromosome after mutation operator.
        """
        for i, _ in enumerate(chromosome):
            if np.random.uniform() < self._mutation_prob:
                chromosome[i] = self._rand_function(
                    low=self._gene_intervals[i][0], high=self._gene_intervals[i][1]
                )
        return chromosome

    def _parents_offspring_tournament(
        self, parents_ix: List[int], offsprings: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tournament amongst two parents and their offsprings.

        Applied when :attr:`new_pop_type` is "tournament" or
        "random". Returns two best chromosomes amongst two parents
        and their two offsprings.

        :param parents_ix: parents indexes in population
        :param offsprings: offsprings chromosomes
        :return: two best chromosomes with best fitness values
        """
        chromosome_arr = [
            self._population[parents_ix[0]],
            self._population[parents_ix[1]],
            offsprings[0],
            offsprings[1],
        ]
        fitness_arr = [
            self._fitness_values[parents_ix[0]],
            self._fitness_values[parents_ix[1]],
            self._fitness_function(offsprings[0]),
            self._fitness_function(offsprings[1]),
        ]

        if self._objective_goal == "maximize":
            two_best_ix = np.argsort(fitness_arr)[-2:]
            # best chromosome is two_best_ix[1] and second best is two_best_ix[0]
        else:
            two_best_ix = np.argsort(fitness_arr)[:2]
            # best chromosome is two_best_ix[0] and second best is two_best_ix[1]

        return chromosome_arr[two_best_ix[0]], chromosome_arr[two_best_ix[1]]

    def _create_new_population(self) -> None:
        """Fill new population with random chromosomes.

        During the phase of creating new population when
        2* :attr:`n_pairs` + :attr:`n_elite` is smaller than
        :attr:`population_size` then it is necessary to fill empty
        slots in a new population. Because there was not enough new
        individuals created during the reproduction (crossover,
        mutation).
        """
        for i in range(self._n_pairs * 2 + self._n_elite, self._population_size):
            self._new_population[i] = self._population[
                np.random.randint(low=0, high=self._population_size - 1)
            ]

        self._population = self._new_population  # type: ignore

    def _get_final_results(self) -> None:
        """Get results after :meth:`simulate() <simulate>`.

        Find best chromosome and its fitness value in population.
        """
        self._evaluate_population()

        if self._objective_goal == "maximize":
            self.best_fitness = np.amax(self._fitness_values)
            best_index = np.argmax(self._fitness_values)
        else:
            self.best_fitness = np.amin(self._fitness_values)
            best_index = np.argmin(self._fitness_values)

        self.best_chromosome = self._population[best_index]
