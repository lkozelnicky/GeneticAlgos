Advanced usage
==============

Genetic algorithms in their original structure have a very simple workflow: create population,
selection, crossover, mutation, new population, and repeat it over and over again. But as genetic
algorithms started to receive a lot of academic attention, they started to be more and more complex
with numerous tweaks in every part of the workflow.

GeneticAlgos implements some algorithm tweaks and modifications which can be adapted and easily tested.
It is important to say that even GeneticAlgos has default parameters which might suit for generic use,
but default values are definitely not suitable for all cases. For this reason, it is possible
to modify genetic algorithm simulation with the desired parameters.

Change default workflow methods
________________________________

In GeneticAlgos, we can tweak different methods for GA workflow building blocks:

* :ref:`selection`
* :ref:`crossover`
* :ref:`mutation`
* :ref:`creating_new_population`

Change parameters during instantiation
______________________________________

When instantiating GeneticAlgo objects, we can modify the following parameters:

.. _gene_intervals:

* **gene_intervals**: is a mandatory parameter when creating a GeneticAlgo object. Parameter expect a 2-dimensional ``numpy.ndarray`` which specifies:

    * chromosome length
    * chromosome encoding
    * value range for each gene - lower bound is inclusive and upper bound is exclusive. Example: integer chromosome with gene range [0, 3] can have values 0, 1, 2.

.. code-block:: python

    import geneticalgos as ga
    import numpy as np

    # 10 gene binary chromosome
    binary_chromosome = np.array([[0, 2]] * 10)
    bin_chromosome_type = "int"

    # 5 gene int chromosome and every gene values from -10 to 10
    num_chromosome_1 = np.array([[-10, 11]] * 5)
    num_chromosome_1_type = "int"

    # 4 gene float chromosome:
    #  - gene 1 values from 0 to 1
    #  - gene 2 - 4 have can have values from 0 to 5
    num_chromosome_2 = np.array([[0, 1], [0, 5], [0, 5], [0, 5]])
    num_chromosome_2_type = "float"

    ga_model = ga.GeneticAlgo(
        fitness_function=sum,
        gene_intervals=binary_chromosome,
        chromosome_type=bin_chromosome_type
    )

* **objective_goal:** optional parameter specifies whether our fitness function is trying to find a maximum or minimum value for a given problem. Supported values are only ``maximize`` and ``minimize``. The default value is ``maximize`` and can easily be changed to ``minimize``.

.. code-block:: python

    import geneticalgos as ga
    import numpy as np

    ga_model = ga.GeneticAlgo(
        fitness_function=sum,
        gene_intervals=np.array([[0, 10]] * 5),
        objective_goal="minimize",    # default value is "maximize"
    )

* **population_size:** specifies population size. The default value is 100 chromosomes (solutions). It can be changed:

.. code-block:: python

    import geneticalgos as ga
    import numpy as np

    ga_model = ga.GeneticAlgo(
        fitness_function=sum,
        gene_intervals=np.array([[0, 10]] * 5),
        population_size=30,   # default is 100
    )

* **chromosome_type:** specifies the type of :ref:`encoding`. Supported encodings are ``int`` and ``float``. The default value is ``float``. It can be changed:

.. code-block:: python

    import geneticalgos as ga
    import numpy as np

    ga_model = ga.GeneticAlgo(
        fitness_function=sum,
        gene_intervals=np.array([[0, 10]] * 5),
        chromosome_type="int",   # default is float
    )


Simulation parameters
_____________________

At the moment, GeneticAlgos supports only one end criterion for simulation and it is a finite
number of iterations (generation cycles). Evolution ends when the ``n_iterations`` parameter is reached.
The default value is 100. The number of iterations can be changed in the simulate method in the
form of a parameter to this method:

.. code-block:: python

    ga_model.simulate(n_iterations=250)   # default is 100

Miscellaneous parameters
________________________

* **n_pairs:** specifies the number of pairs selected to reproduce. By default, it is 1/3 of the population size. We can change the number of pairs with the following command:

.. code-block:: python

    ga_model.n_pairs = 40

.. hint:: When the number of ``n_pairs * 2 + n_elite`` is greater than ``population_size`` then **ValueError** is raised with guidance to which interval is allowed for ``n_pairs``.

.. _explore_results:

Explore simulation results
__________________________

The best solution fitness value and chromosome gene values can be accessed after ``simulate()`` is finished.

.. code-block:: python

    import geneticalgos as ga
    import numpy as np

    # trivial GA model:
    #   search maximum sum in float chromosome
    ga_model = ga.GeneticAlgo(
        fitness_function=sum,
        gene_intervals=np.array([[0, 100]] * 10),
    )
    ga_model.simulate()

    # best chrosomome gene values
    print(ga_model.best_chromosome)

    # fitness value for best chromosome
    print(ga_model.best_fitness)


It is possible to visualize the evolving best fitness value during the evolution process.
We use ``best_chromosomes_history`` attribute which stores the best fitness values after each
generation cycle.

First we have to install ``matplotlib`` library.

.. code-block:: bash

   $ pip install matplotlib

``best_chromosomes_history`` is a ``List`` which stores the best fitness values after each
generation cycle.

.. code-block:: python

    import matplotlib.pyplot as plt

    plt.plot(ga_model.best_chromosomes_history)
    plt.show()




