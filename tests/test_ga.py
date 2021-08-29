import operator
from unittest.mock import patch

import numpy as np
import pytest

from geneticalgos import ga
from geneticalgos import exceptions


@pytest.fixture
def genetic_algo():
    return ga.GeneticAlgo(
        population_size=4,
        fitness_function=sum,
        gene_intervals=np.array([[0, 10], [0, 10]]),
    )


def test_constructor(genetic_algo):
    assert isinstance(genetic_algo, ga.GeneticAlgo)


def test_constructor_exceptions():
    with pytest.raises(exceptions.FitnessFunctionNotCallable):
        ga.GeneticAlgo(fitness_function=20, gene_intervals=np.array([]))
    with pytest.raises(exceptions.InvalidObjectiveGoal):
        ga.GeneticAlgo(
            fitness_function=sum,
            gene_intervals=np.array([[0, 10], [0, 10]]),
            objective_goal=100,
        )
    with pytest.raises(TypeError, match=r".*population_size type must be int.*"):
        ga.GeneticAlgo(
            fitness_function=sum,
            gene_intervals=np.array([[0, 10], [0, 10]]),
            population_size="100",
        )
    with pytest.raises(TypeError, match=r".*population_size type must be int.*"):
        ga.GeneticAlgo(
            fitness_function=sum,
            gene_intervals=np.array([[0, 10], [0, 10]]),
            population_size=100.0,
        )
    with pytest.raises(ValueError, match=r".*population_size must be greater than 3.*"):
        ga.GeneticAlgo(
            fitness_function=sum,
            gene_intervals=np.array([[0, 10], [0, 10]]),
            population_size=-10,
        )
    with pytest.raises(ValueError, match=r".*population_size must be greater than 3.*"):
        ga.GeneticAlgo(
            fitness_function=sum,
            gene_intervals=np.array([[0, 10], [0, 10]]),
            population_size=3,
        )
    with pytest.raises(exceptions.InvalidChromosomeType):
        ga.GeneticAlgo(
            chromosome_type="str",
            fitness_function=sum,
            gene_intervals=np.array([[0, 10], [0, 10]]),
        )
    with pytest.raises(TypeError, match=r".*chromosome_type must be str.*"):
        ga.GeneticAlgo(
            chromosome_type=100,
            fitness_function=sum,
            gene_intervals=np.array([[0, 10], [0, 10]]),
        )
    with pytest.raises(TypeError, match=r".*gene_intervals must be numpy.ndarray.*"):
        ga.GeneticAlgo(
            fitness_function=sum,
            gene_intervals=[[0, 10], [0, 10]],
        )
    with pytest.raises(
        exceptions.InvalidGeneIntervalsShape,
        match=r".*not valid gene_intervals shape.*",
    ):
        ga.GeneticAlgo(
            fitness_function=sum,
            gene_intervals=np.array([[[0, 10], [0, 10]], [[0, 10], [0, 10]]]),
        )
    with pytest.raises(
        exceptions.InvalidGeneIntervalsShape,
        match=r".*not valid gene_intervals shape.*",
    ):
        ga.GeneticAlgo(
            fitness_function=sum,
            gene_intervals=np.array([[0, 10, 100], [0, 10, 100]]),
        )


def test_fitness_returns_none():
    def custom_fitness_none(chromosome):
        sum(chromosome)

    ga_test = ga.GeneticAlgo(custom_fitness_none, np.array([[0, 10], [0, 10]]))
    ga_test._population = np.array([[0.5, 0.3], [0.085, 0.23]])
    with pytest.raises(exceptions.FitnessFunctionReturnsNone):
        ga_test._evaluate_population()


def test_selection_type_prop(genetic_algo):
    genetic_algo.selection_type = "roulette_fitness"
    assert genetic_algo.selection_type == "roulette_fitness"
    assert genetic_algo._selection_type == "roulette_fitness"
    genetic_algo.selection_type = "roulette_rank"
    genetic_algo.selection_type = "tournament"

    with pytest.raises(TypeError, match=r".*selection_type must be str.*"):
        genetic_algo.selection_type = 100

    with pytest.raises(ValueError, match=r".*supported selection types.*"):
        genetic_algo.selection_type = "unknown"


def test_selection_k_prop(genetic_algo):
    genetic_algo.selection_type = "tournament"
    genetic_algo.selection_k = 5
    assert genetic_algo.selection_k == 5
    assert genetic_algo._selection_k == 5

    with pytest.raises(TypeError, match=r".*selection_k size must be int.*"):
        genetic_algo.selection_k = 100.0
    with pytest.raises(
        ValueError, match=r".*selection_k size must be greater than zero.*"
    ):
        genetic_algo.selection_k = -10

    genetic_algo.selection_k_max = 20
    with pytest.raises(
        ValueError, match=r".*selection_k size must be greater than zero.*"
    ):
        genetic_algo.selection_k = 25

    genetic_algo.selection_type = "roulette_rank"
    with pytest.raises(ValueError, match=r".*selection_k size is supported for.*"):
        genetic_algo.selection_k = 4


def test_crossover_type_prop(genetic_algo):
    genetic_algo.crossover_type = "one_point"
    assert genetic_algo.crossover_type == "one_point"
    assert genetic_algo._crossover_type == "one_point"
    genetic_algo.crossover_type = "two_points"
    genetic_algo.crossover_type = "uniform"

    with pytest.raises(TypeError, match=r".*crossover_type must be str.*"):
        genetic_algo.crossover_type = 100
    with pytest.raises(ValueError, match=r".*supported crossover types.*"):
        genetic_algo.crossover_type = "unknown"


def test_crossover_prob_prop(genetic_algo):
    genetic_algo.crossover_prob = 0.5
    assert genetic_algo.crossover_prob == 0.5
    assert genetic_algo._crossover_prob == 0.5
    genetic_algo.crossover_prob = 0.0
    genetic_algo.crossover_prob = 1.0

    with pytest.raises(TypeError, match=r".*crossover_prob must be float.*"):
        genetic_algo.crossover_prob = "0.5"
    with pytest.raises(ValueError, match=r".*crossover_prob must be from.*"):
        genetic_algo.crossover_prob = 1.1
    with pytest.raises(ValueError, match=r".*crossover_prob must be from.*"):
        genetic_algo.crossover_prob = -0.5


def test_mutation_prop(genetic_algo):
    genetic_algo.mutation_prob = 0.5
    assert genetic_algo.mutation_prob == 0.5
    assert genetic_algo._mutation_prob == 0.5

    genetic_algo.mutation_prob = 0.0
    genetic_algo.mutation_prob = 1.0

    with pytest.raises(TypeError, match=r".*mutation_prob must be float.*"):
        genetic_algo.mutation_prob = "0.5"
    with pytest.raises(TypeError, match=r".*mutation_prob must be float.*"):
        genetic_algo.mutation_prob = 0
    with pytest.raises(ValueError, match=r".*mutation_prob must be from.*"):
        genetic_algo.mutation_prob = 1.1
    with pytest.raises(ValueError, match=r".*mutation_prob must be from.*"):
        genetic_algo.mutation_prob = -0.5


def test_new_pop_type_prop(genetic_algo):
    genetic_algo.new_pop_type = "always_offsprings"
    assert genetic_algo.new_pop_type == "always_offsprings"
    assert genetic_algo._new_pop_type == "always_offsprings"
    genetic_algo.new_pop_type = "tournament"

    with pytest.raises(TypeError, match=r".*new_pop_type must be str.*"):
        genetic_algo.new_pop_type = 100
    with pytest.raises(ValueError, match=r".*supported new_pop_type.*"):
        genetic_algo.new_pop_type = "unknown"


@pytest.fixture
def genetic_algo_size_100():
    return ga.GeneticAlgo(
        population_size=100,
        fitness_function=sum,
        gene_intervals=np.array([[0, 10], [0, 10]]),
    )


def test_n_pairs_prop(genetic_algo_size_100):
    genetic_algo_size_100.n_pairs = 20
    assert genetic_algo_size_100.n_pairs == 20
    assert genetic_algo_size_100._n_pairs == 20

    genetic_algo_size_100.n_pairs = 1
    # Expecting default n_elite = 2
    genetic_algo_size_100.n_pairs = 49

    with pytest.raises(TypeError, match=r".*n_pairs must be int.*"):
        genetic_algo_size_100.n_pairs = "48"
    with pytest.raises(TypeError, match=r".*n_pairs must be int.*"):
        genetic_algo_size_100.n_pairs = 48.0
    with pytest.raises(ValueError, match=r".*n_pairs must be int from interval.*"):
        genetic_algo_size_100.n_pairs = 0
    with pytest.raises(ValueError, match=r".*n_pairs must be int from interval.*"):
        genetic_algo_size_100.n_pairs = -10
    with pytest.raises(ValueError, match=r".*n_pairs must be int from interval.*"):
        genetic_algo_size_100.n_pairs = 51


def test_n_elite_prop(genetic_algo_size_100):
    genetic_algo_size_100.n_elite = 20
    assert genetic_algo_size_100.n_elite == 20
    assert genetic_algo_size_100._n_elite == 20
    # Expecting default n_pairs: 100 / 3 = 33
    genetic_algo_size_100.n_elite = 34
    genetic_algo_size_100.n_elite = 1
    # genetic_algo_size_100.n_elite = 0

    with pytest.raises(TypeError, match=r".*n_elite must be int.*"):
        genetic_algo_size_100.n_elite = "20"
    with pytest.raises(TypeError, match=r".*n_elite must be int.*"):
        genetic_algo_size_100.n_elite = 12.0
    with pytest.raises(ValueError, match=r".*n_elite must be int from interval.*"):
        genetic_algo_size_100.n_elite = -10
    with pytest.raises(ValueError, match=r".*n_elite must be int from interval.*"):
        genetic_algo_size_100.n_elite = 101


def test_n_pairs_n_elite_combinations(genetic_algo_size_100):
    with pytest.raises(ValueError, match=r".*n_pairs must be int from interval.*"):
        genetic_algo_size_100.n_elite = 1
        genetic_algo_size_100.n_pairs = 50
    with pytest.raises(ValueError, match=r".*n_elite must be int from interval.*"):
        genetic_algo_size_100.n_pairs = 40
        genetic_algo_size_100.n_elite = 21


@pytest.mark.parametrize(
    "gene_intervals, chromosome_type, population_size, low, high",
    [
        ([[0, 10], [-10, 30]], "int", 100, -10, 30),
        ([[1.5, 10], [2.5, 20]], "float", 500, 1.5, 20),
        ([[1.5, 9.8], [1.6, 9], [1.5, 9.8]], "float", 100, 1.5, 9.8),
        ([[-1.5, 10], [-10.5, 8]], "float", 250, -10.5, 10),
    ],
)
def test_create_initial_population(
    gene_intervals, chromosome_type, population_size, low, high
):
    ga_test = ga.GeneticAlgo(
        fitness_function=sum,
        gene_intervals=np.array(gene_intervals),
        chromosome_type=chromosome_type,
        population_size=population_size,
    )
    ga_test._create_initial_population()
    assert len(ga_test._population[0]) == len(gene_intervals)
    assert len(ga_test._population) == population_size
    assert all([(low <= i < high) for row in ga_test._population for i in row])


def test_evaluate_population(genetic_algo):
    genetic_algo._population = np.array(
        [[0.5, 0.3], [0.085, 0.23], [9.408, 1.24], [0.05, 0.15]]
    )
    genetic_algo._evaluate_population()
    assert len(genetic_algo._population) == len(genetic_algo._fitness_values)
    assert np.allclose(genetic_algo._fitness_values, [0.8, 0.315, 10.648, 0.2])
    assert genetic_algo.best_chromosomes_history[-1] == 10.648
    genetic_algo._population = np.array(
        [[0.2, 8.265], [5.65, 0.87], [5.49, 1.89564], [0.1, 4.4]]
    )
    genetic_algo._evaluate_population()
    assert np.allclose(genetic_algo._fitness_values, [8.465, 6.52, 7.3856, 4.5])
    assert genetic_algo.best_chromosomes_history[-1] == 8.465


@pytest.mark.parametrize(
    "population_size, population, n_elite, n_pairs, start_pos",
    [
        (4, [[5.5, 0.3], [0.085, 0.23], [9.408, 1.24], [0.05, 0.15]], 2, 1, 2),
        (
            5,
            [[5.5, 0.3], [0.085, 0.23], [0.56, 3.1], [9.408, 1.24], [0.05, 0.15]],
            2,
            1,
            2,
        ),
        (
            6,
            [
                [5.5, 0.3],
                [0.085, 0.23],
                [0.6, 3],
                [0.5, 1.5],
                [9.408, 1.24],
                [0.05, 0.15],
            ],
            2,
            2,
            4,
        ),
        (
            8,
            [
                [5.5, 0.3],
                [1, 3],
                [0.1, 1],
                [2, 1],
                [9.408, 1.24],
                [2, 2],
                [2, 1],
                [1, 1],
            ],
            2,
            3,
            6,
        ),
    ],
)
def test_add_elite(population_size, population, n_elite, n_pairs, start_pos):
    ga_test = ga.GeneticAlgo(
        fitness_function=sum,
        gene_intervals=np.array([[0, 10], [-10, 30]]),
        population_size=population_size,
    )
    ga_test._population = np.array(population)
    ga_test.n_elite = n_elite
    ga_test.n_pairs = n_pairs
    ga_test._evaluate_population()
    ga_test._new_population = np.zeros(shape=[population_size, 2])
    ga_test._add_elite()
    assert np.allclose(ga_test._new_population[start_pos], [5.5, 0.3])
    assert np.allclose(ga_test._new_population[start_pos + 1], [9.408, 1.24])


def test_random_selection(genetic_algo):
    genetic_algo._population = np.array(
        [
            [9.408, 1.24],
            [0.05, 0.15],
            [4.55, 9.3],
            [14.8, 8.59],
        ]
    )
    genetic_algo._fitness_values = [10.648, 0.2, 13.85, 23.39]
    for i in range(100):
        selected_parents = genetic_algo._random_selection()
        assert isinstance(selected_parents, list)
        assert len(selected_parents) == genetic_algo.n_pairs * 2
        assert all(isinstance(i, (np.int64, int)) for i in selected_parents)


@pytest.mark.parametrize(
    "random_numbers, fitness_values, obj_goal, return_value",
    [
        ([0.19, 0.58], np.array([7.7, 15.7, 17.6, 4.8]), "maximize", [1, 2]),
        ([0.19, 0.58], np.array([7.7, 15.7, 17.6, 4.8]), "minimize", [0, 2]),
        ([0.11, 0.9], np.array([7.7, 15.7, 17.6, 4.8]), "maximize", [0, 3]),
        ([0.29, 0.65], np.array([7.7, 15.7, 17.6, 4.8]), "minimize", [1, 2]),
    ],
)
def test_roulette_fitness_selection(
    genetic_algo, random_numbers, fitness_values, obj_goal, return_value
):
    with patch("numpy.random.uniform", side_effect=random_numbers):
        genetic_algo.selection_type = "roulette_fitness"
        genetic_algo._objective_goal = obj_goal
        genetic_algo._fitness_values = fitness_values
        selected_parents = genetic_algo._roulette_fitness_selection()
        assert isinstance(selected_parents, list)
        assert all(isinstance(i, (np.int64, int)) for i in selected_parents)
        assert len(selected_parents) == (genetic_algo.n_pairs * 2)
        assert selected_parents == return_value


@pytest.mark.parametrize(
    "random_numbers, fitness_values, obj_goal, return_value",
    [
        ([0.19, 0.58], np.array([7.7, 15.7, 17.6, 4.8]), "maximize", [0, 1]),
        ([0.19, 0.58], np.array([7.7, 15.7, 17.6, 4.8]), "minimize", [1, 0]),
        ([0.11, 0.9], np.array([7.7, 15.7, 17.6, 4.8]), "maximize", [0, 2]),
        ([0.29, 0.65], np.array([7.7, 15.7, 17.6, 4.8]), "minimize", [1, 3]),
    ],
)
def test_roulette_rank_selection(
    genetic_algo, random_numbers, fitness_values, obj_goal, return_value
):
    with patch("numpy.random.uniform", side_effect=random_numbers):
        genetic_algo.selection_type = "roulette_rank"
        genetic_algo._objective_goal = obj_goal
        genetic_algo._fitness_values = fitness_values
        selected_parents = genetic_algo._roulette_rank_selection()
        assert isinstance(selected_parents, list)
        assert all(isinstance(i, (np.int64, int)) for i in selected_parents)
        assert len(selected_parents) == (genetic_algo.n_pairs * 2)
        assert selected_parents == return_value


@pytest.mark.parametrize(
    "obj_goal, r_value",
    [("maximize", [2, 0, 6, 5]), ("minimize", [4, 4, 7, 1])],
)
def test_tournament_selection(obj_goal, r_value):
    ga_test = ga.GeneticAlgo(
        fitness_function=sum,
        gene_intervals=np.array([[0, 10], [-10, 30]]),
        objective_goal=obj_goal,
        population_size=10,
    )
    ga_test._fitness_values = np.array(
        [
            20.76587112,
            7.87875307,
            39.62627714,
            28.51930456,
            3.89731135,
            27.81629036,
            35.20556279,
            11.83980045,
            23.81584026,
            21.13863493,
        ]
    )
    ga_test.n_pairs = 2
    ga_test.selection_type = "tournament"
    ga_test.selection_k = 3
    random_randint = [4, [1, 2], 0, [7, 4], 7, [6, 3], 1, [0, 5]]
    with patch("numpy.random.randint", side_effect=random_randint):
        selected_parents = ga_test._tournament_selection()
        assert selected_parents == r_value
        assert isinstance(selected_parents, list)
        assert len(selected_parents) == (ga_test.n_pairs * 2)
        assert all(isinstance(i, int) for i in selected_parents)


def test_random_crossover(genetic_algo):
    genetic_algo._population = np.array(
        [
            [9.408, 1.24],
            [0.05, 0.15],
            [4.55, 9.3],
            [14.8, 8.59],
        ]
    )
    parent_1 = genetic_algo._population[0]
    parent_2 = genetic_algo._population[1]
    for i in range(100):
        offspring_1, offspring_2 = genetic_algo._random_crossover(parent_1, parent_2)
        assert isinstance(offspring_1, np.ndarray)
        assert len(offspring_1) == 2
        assert isinstance(offspring_2, np.ndarray)
        assert len(offspring_2) == 2


@pytest.mark.parametrize(
    "crossover_prob, crossover_point, r_value_1, r_value_2",
    [
        (0.8, 0, np.array([1.8, 1.1, 1.9, 5.8]), np.array([7.7, 5.7, 17.6, 4.2])),
        (0.5, 4, np.array([7.7, 5.7, 17.6, 4.2]), np.array([1.8, 1.1, 1.9, 5.8])),
        (0.75, 2, np.array([7.7, 5.7, 1.9, 5.8]), np.array([1.8, 1.1, 17.6, 4.2])),
        (0.89, 1, np.array([7.7, 1.1, 1.9, 5.8]), np.array([1.8, 5.7, 17.6, 4.2])),
        (0.91, 2, np.array([7.7, 5.7, 17.6, 4.2]), np.array([1.8, 1.1, 1.9, 5.8])),
    ],
)
def test_one_point_crossover(
    genetic_algo, crossover_prob, crossover_point, r_value_1, r_value_2
):
    with patch("numpy.random.uniform", return_value=crossover_prob):
        with patch("numpy.random.randint", return_value=crossover_point):
            genetic_algo.crossover_prob = 0.9
            parent_1 = np.array([7.7, 5.7, 17.6, 4.2])
            parent_2 = np.array([1.8, 1.1, 1.9, 5.8])
            offspring_1, offspring_2 = genetic_algo._one_point_crossover(
                parent_1, parent_2
            )
            assert np.allclose(offspring_1, r_value_1)
            assert isinstance(offspring_1, np.ndarray)
            assert np.allclose(offspring_2, r_value_2)
            assert isinstance(offspring_2, np.ndarray)


@pytest.mark.parametrize(
    "crossover_prob, crossover_points, r_value_1, r_value_2",
    [
        (0.8, [0, 1], np.array([1.8, 5.7, 17.6, 4.2]), np.array([7.7, 1.1, 1.9, 5.8])),
        (0.2, [2, 4], np.array([7.7, 5.7, 1.9, 5.8]), np.array([1.8, 1.1, 17.6, 4.2])),
        (0.6, [1, 3], np.array([7.7, 1.1, 1.9, 4.2]), np.array([1.8, 5.7, 17.6, 5.8])),
        (0.89, [3, 4], np.array([7.7, 5.7, 17.6, 5.8]), np.array([1.8, 1.1, 1.9, 4.2])),
        (0.91, [1, 3], np.array([7.7, 5.7, 17.6, 4.2]), np.array([1.8, 1.1, 1.9, 5.8])),
    ],
)
def test_two_points_crossover(
    genetic_algo, crossover_prob, crossover_points, r_value_1, r_value_2
):
    with patch("numpy.random.uniform", return_value=crossover_prob):
        with patch("geneticalgos.ga.sample", return_value=crossover_points):
            genetic_algo.crossover_prob = 0.9
            parent_1 = np.array([7.7, 5.7, 17.6, 4.2])
            parent_2 = np.array([1.8, 1.1, 1.9, 5.8])
            offspring_1, offspring_2 = genetic_algo._two_points_crossover(
                parent_1, parent_2
            )
            assert np.allclose(offspring_1, r_value_1)
            assert isinstance(offspring_1, np.ndarray)
            assert np.allclose(offspring_2, r_value_2)
            assert isinstance(offspring_2, np.ndarray)


@pytest.mark.parametrize(
    "random_uniform, r_value_1, r_value_2",
    [
        (
            [0.8, 0.2, 0.3, 0.8, 0.4],
            np.array([7.7, 5.7, 1.9, 4.2]),
            np.array([1.8, 1.1, 17.6, 5.8]),
        ),
        (
            [0.6, 0.7, 0.49, 0.51, 0.78],
            np.array([1.8, 5.7, 1.9, 5.8]),
            np.array([7.7, 1.1, 17.6, 4.2]),
        ),
        (
            [0.5, 0.1, 0.8, 0.2, 0.9],
            np.array([7.7, 1.1, 17.6, 5.8]),
            np.array([1.8, 5.7, 1.9, 4.2]),
        ),
        (
            [0.91, 0.1, 0.8, 0.2, 0.9],
            np.array([7.7, 5.7, 17.6, 4.2]),
            np.array([1.8, 1.1, 1.9, 5.8]),
        ),
    ],
)
def test_uniform_crossover(genetic_algo, random_uniform, r_value_1, r_value_2):
    with patch("numpy.random.uniform", side_effect=random_uniform):
        genetic_algo.crossover_prob = 0.9
        parent_1 = np.array([7.7, 5.7, 17.6, 4.2])
        parent_2 = np.array([1.8, 1.1, 1.9, 5.8])
        offspring_1, offspring_2 = genetic_algo._uniform_crossover(parent_1, parent_2)
        assert np.allclose(offspring_1, r_value_1)
        assert isinstance(offspring_1, np.ndarray)
        assert np.allclose(offspring_2, r_value_2)
        assert isinstance(offspring_2, np.ndarray)


def test_mutation(genetic_algo):
    genetic_algo.mutation_prob = 0.2
    random_numbers = [0.1, 0.5]
    with patch("numpy.random.uniform", side_effect=random_numbers):
        new_chromosome = genetic_algo._mutation(np.array([1.8, 5.7]))
        # only first gene is mutating
        assert new_chromosome[0] != 1.8 and 0 < new_chromosome[0] < 10
        assert new_chromosome[1] == 5.7
        assert isinstance(new_chromosome, np.ndarray)


@pytest.mark.parametrize(
    "obj_goal, parents_ix, offspring_1, offspring_2, r_value_1, r_value_2",
    [
        ("maximize", [0, 1], [0.5, 0.23], [0.085, 0.3], [0.5, 0.23], [0.5, 0.3]),
        ("maximize", [1, 3], [0.085, 0.15], [0.05, 0.23], [0.05, 0.23], [0.085, 0.23]),
        ("minimize", [0, 1], [0.5, 0.23], [0.085, 0.3], [0.085, 0.23], [0.085, 0.3]),
        ("minimize", [1, 3], [0.085, 0.15], [0.05, 0.23], [0.05, 0.15], [0.085, 0.15]),
    ],
)
def test_parents_offspring_tournament(
    obj_goal, parents_ix, offspring_1, offspring_2, r_value_1, r_value_2
):
    ga_test = ga.GeneticAlgo(
        fitness_function=sum,
        gene_intervals=np.array([[0, 10], [-10, 30]]),
        population_size=4,
        objective_goal=obj_goal,
    )
    ga_test._population = np.array(
        [[0.5, 0.3], [0.085, 0.23], [9.408, 1.24], [0.05, 0.15]]
    )
    ga_test._evaluate_population()
    value_1, value_2 = ga_test._parents_offspring_tournament(
        parents_ix=parents_ix, offsprings=[np.array(offspring_1), np.array(offspring_2)]
    )
    assert np.allclose(value_1, r_value_1)
    assert isinstance(value_1, np.ndarray)
    assert np.allclose(value_2, r_value_2)
    assert isinstance(value_2, np.ndarray)


def test_create_new_population(genetic_algo):
    ga_test = ga.GeneticAlgo(
        fitness_function=sum,
        gene_intervals=np.array([[0, 10], [-10, 30]]),
        population_size=6,
    )
    ga_test._population = np.array(
        [
            [0.5, 8.3],
            [0.085, 0.23],
            [9.408, 1.24],
            [0.05, 0.15],
            [4.55, 9.3],
            [14.8, 8.59],
        ]
    )
    ga_test.n_elite = 2
    ga_test.n_pairs = 1
    ga_test._new_population = np.array(
        [[0.5, 0.23], [9.408, 0.15], [14.8, 8.59], [4.55, 9.3], [0.0, 0.0], [0.0, 0.0]]
    )
    ga_test._create_new_population()
    assert np.all(ga_test._population)
    assert len(ga_test._population) == 6
    assert isinstance(ga_test._population, np.ndarray)


@pytest.mark.parametrize(
    "obj_goal, best_fitness, best_chromosome",
    [("maximize", 10.648, [9.408, 1.24]), ("minimize", 0.2, [0.05, 0.15])],
)
def test_get_final_results(obj_goal, best_fitness, best_chromosome):
    ga_test = ga.GeneticAlgo(
        sum,
        np.array([[0, 10], [0, 10], [0, 10], [0, 10]]),
        objective_goal=obj_goal,
        chromosome_type="float",
        population_size=4,
    )
    ga_test._population = np.array(
        [[5.5, 0.3], [0.085, 0.23], [9.408, 1.24], [0.05, 0.15]]
    )
    ga_test._get_final_results()
    assert ga_test.best_fitness == best_fitness
    assert np.allclose(ga_test.best_chromosome, np.array(best_chromosome))


def test_n_iterations_exceptions(genetic_algo):
    with pytest.raises(TypeError, match=r".*n_iterations type must be int.*"):
        genetic_algo.simulate(n_iterations="100")
    with pytest.raises(TypeError, match=r".*n_iterations type must be int.*"):
        genetic_algo.simulate(n_iterations=20.0)
    with pytest.raises(exceptions.InvalidNumberOfIterations):
        genetic_algo.simulate(n_iterations=0)
    with pytest.raises(exceptions.InvalidNumberOfIterations):
        genetic_algo.simulate(n_iterations=-10)


@pytest.mark.parametrize(
    "obj_goal, comparison_operator, new_pop_type, r_value",
    [
        ("maximize", operator.gt, "tournament", 55),
        ("minimize", operator.lt, "tournament", 5),
        ("maximize", operator.gt, "always_offsprings", 55),
        ("minimize", operator.lt, "always_offsprings", 5),
        ("maximize", operator.gt, "random", 55),
        ("minimize", operator.lt, "random", 5),
    ],
)
def test_simulate(obj_goal, comparison_operator, new_pop_type, r_value):
    ga_test = ga.GeneticAlgo(
        sum,
        np.array([[0, 10]] * 6),
        objective_goal=obj_goal,
        chromosome_type="float",
        population_size=100,
    )
    ga_test.new_pop_type = new_pop_type
    ga_test.simulate()
    assert comparison_operator(ga_test.best_fitness, r_value)
    assert np.all(ga_test._population)


@pytest.mark.parametrize(
    "obj_goal, comparison_operator, new_pop_type, r_value",
    [
        ("maximize", operator.gt, "tournament", 245),
        ("minimize", operator.lt, "tournament", -55),
        ("maximize", operator.gt, "always_offsprings", 245),
        ("minimize", operator.lt, "always_offsprings", -55),
        ("maximize", operator.gt, "random", 245),
        ("minimize", operator.lt, "random", -55),
    ],
)
def test_simulate_adv(obj_goal, comparison_operator, new_pop_type, r_value):
    # equation  10a - 1b - 6c + 4d + 12e
    values = [10, -1, -6, 4, 12]

    def custom_fit(chromosome):
        return sum(x * y for x, y in zip(chromosome, values))

    ga_test = ga.GeneticAlgo(
        custom_fit,
        np.array([[0, 10]] * 5),
        objective_goal=obj_goal,
        chromosome_type="float",
        population_size=100,
    )
    ga_test.new_pop_type = new_pop_type
    ga_test.simulate()
    assert comparison_operator(ga_test.best_fitness, r_value)
    assert np.all(ga_test._population)
