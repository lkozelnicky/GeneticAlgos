Getting started
===============

Python version support
______________________

GeneticAlgos should be used with latest version of Python, and it is tested
on Python 3.7, 3.8 and 3.9.

Installation
____________

``geneticalgos`` is available on `PYPI <https://pypi.python.org/pypi/GeneticAlgos/>`_.
Install with ``pip``:

.. code-block:: bash

   $ pip install geneticalgos

Usage
_____

Every problem we are trying to solve might be different and may require a different approach.
When using GeneticAlgos, there are some mandatory and some optional steps for successful genetic
algorithms application in practical scenarios.

The best way to demonstrate this is to show GeneticAlgos in use on a real example and illustrate
all the steps required.

For example, we can define ``-(x^2) + y`` maximization function that takes input variables from
range (-10, 10) in float numbers. This function has optima at f(0, 9.999) = 9.999

1. Create custom :ref:`fitness_function` which returns the numerical value that represents the suitability of the given solution.

.. code-block:: python

    import geneticalgos as ga
    import numpy as np

    def custom_fitness_function(chromosome):
        return -(chromosome[0] ** 2) + chromosome[1]

.. warning:: When ``fitness_function`` does not return anything (returns ``None`` python default) then the``FitnessFunctionReturnsNone`` exception is raised.

2. :ref:`Gene_intervals <gene_intervals>` - specify the length and encoding of the chromosome together with the value range for each gene.

.. code-block:: python

    gene_intervals = np.array([[-10, 10], [-10, 10]])


3. Create the genetic algorithms model.

.. code-block:: python

    ga_model = ga.GeneticAlgo(
        fitness_function=custom_fitness_function,
        gene_intervals=gene_intervals,
    )


4. The good news is that GeneticAlgos has default values for almost all parameters, but when it is necessary, we can change them very quickly - :doc:`Advanced usage <advanced>`.

5. Start genetic **simulation**.

.. code-block:: python

   ga_model.simulate()


6. :ref:`Explore the results <explore_results>`. We can display the best solution, fitness value. Results will vary because of the stochastic nature of the genetic algorithms.

.. code-block:: python

    print(ga_model.best_chromosome)
    # array([2.99317567e-03, 9.99101672e+00])

    print(ga_model.best_fitness)
    # 9.99100




