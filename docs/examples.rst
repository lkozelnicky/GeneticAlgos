Examples
========

Sometimes it is best to learn from the examples, so let's start...

1. :ref:`example_1`
2. :ref:`example_2`
3. :ref:`example_3`

.. _example_1:

One Max Problem
_______________

The One Max Problem is very simple and widely used in the genetic algorithm community. It is a GA
with binary encoding (genes only 0 or 1), and we want our genetic algorithm to evolve to the
solution where all genes are 1. We use all default values for all GeneticAlgos parameters.

| Brief problem analyses:
| * Binary encoding.
| * Chromosome length - any random positive number, let's have it 30.
| * Fitness function is the sum of all genes and it is a maximization problem.


Code example:

.. code-block:: python

    import geneticalgos as ga
    import numpy as np

    # 30 gene binary chromosome
    binary_chromosome = np.array([[0, 2]] * 30)

    ga_model = ga.GeneticAlgo(
        fitness_function=sum,
        gene_intervals=binary_chromosome,
        chromosome_type="int",
        objective_goal="maximize",    # optional as "maximize" is default
    )

    ga_model.simulate()

    print(ga_model.best_chromosome)
    # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
    # 1. 1. 1. 1. 1. 1.]

.. _example_2:

One Max Problem - specific GA parameters
________________________________________

Let’s take the same One Max Problem as above, where we want to find a solution which has all 1s
in a binary chromosome. But now we want to use specific genetic algorithms parameters:

* Population size 30.
* Tournament selection method with tournament size 5.
* One-point crossover with a crossover probability of 70%.
* Mutation probability of 25%.
* Create a new population always with just offsprings
* Use elitism and move the best 5 chromosomes to the new population automatically.
* Evolution should have at least 120 generation cycles

.. code-block:: python

    import geneticalgos as ga
    import numpy as np

    # 30 gene binary chromosome
    binary_chromosome = np.array([[0, 2]] * 30)

    ga_model = ga.GeneticAlgo(
        fitness_function=sum,
        gene_intervals=binary_chromosome,
        chromosome_type="int",
        population_size=30
    )

    # Tournament selection method with tournament size 5.
    ga_model.selection_type = "tournament"
    ga_model.selection_k = 5

    # One point crossover with crossover probability 70%.
    ga_model.crossover_type = "one_point"
    ga_model.crossover_prob = 0.7

    # Mutation probability 25%.
    ga_model.mutation_prob = 0.25

    # Create new population always with just offsprings
    ga_model.new_pop_type = "always_offsprings"

    #  Use elitism and move best 5 chromosomes to new population automatically.
    ga_model.n_elite = 5

    # Evolution should have at least 120 generation cycles
    ga_model.simulate(n_iterations=130)

    print(ga_model.best_chromosome)
    # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
    # 1. 1. 1. 1. 1. 1.]

.. _example_3:

Minimize function
_________________

So far we have studied examples of maximization problems, but let’s now take a look at a minimization
problem. We define the function ``x^2 + y^2`` searching for minimum values x and y. Both (x and y)
are float numbers within a range (-10, 10). The optimal solution is f(0, 0) = 0. There is one
condition for genetic algorithms parameter - we do not want to use elitism.

.. code-block:: python

    import geneticalgos as ga
    import numpy as np

    def custom_fitness_function(chromosome):
        # x^2 + y^2
        return chromosome[0] ** 2 + chromosome[1] ** 2

    # 30 gene binary chromosome
    chromosome_encoding = np.array([[-10, 10], [-10, 10]])

    ga_model = ga.GeneticAlgo(
        fitness_function=custom_fitness_function,
        gene_intervals=chromosome_encoding,
        objective_goal="minimize",
    )

    # We do not want to use elitism.
    ga_model.n_elite = 0

    ga_model.simulate()

    print(ga_model.best_chromosome)
    # Results will vary because of stochastic nature of algorithms
    # [0.00651341 0.01138962]
