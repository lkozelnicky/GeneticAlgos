GeneticAlgos
============

.. image:: https://github.com/lkozelnicky/GeneticAlgos/workflows/Tests/badge.svg
        :target: https://github.com/lkozelnicky/GeneticAlgos/actions?query=workflow%3ATests+branch%3Amaster

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: https://img.shields.io/pypi/v/GeneticAlgos.svg
    :target: https://pypi.python.org/pypi/GeneticAlgos

.. image:: https://img.shields.io/pypi/l/GeneticAlgos.svg
    :target: https://pypi.python.org/pypi/GeneticAlgos

.. image:: https://img.shields.io/pypi/pyversions/GeneticAlgos.svg
    :target: https://pypi.python.org/pypi/GeneticAlgos

.. image:: https://raw.githubusercontent.com/lkozelnicky/GeneticAlgos/master/docs/_static/GeneticAlgos.png
    :align: center
    :alt: GenticAlgos logo

GeneticAlgos is a simple and powerful Python library for creating genetic algorithms to solve
complex optimization problems. GeneticAlgos is built on NumPy and it is under active development.

- Uses smart defaults for genetic algorithms parameters which are good enough for generic use cases or for testing.
- A simple-to-use API to modify genetic algorithms parameters.
- Lightweight and just one dependency - Numpy.
- Excellent test coverage.
- Tested on Python 3.7, 3.8 and 3.9

Documentation
_____________

Online documentation is available at `https://geneticalgos.readthedocs.io/en/latest/ <https://geneticalgos.readthedocs.io/en/latest/>`_.

The docs include a `introduction to genetic algorithms <https://geneticalgos.readthedocs.io/en/latest/introduction.html>`_,
`examples <https://geneticalgos.readthedocs.io/en/latest/examples.html>`_, `advanced usage <https://geneticalgos.readthedocs.io/en/latest/advanced.html>`_,
and other useful information.

Usage
_____

``geneticalgos`` is available on `PYPI <https://pypi.python.org/pypi/GeneticAlgos/>`_.

Install with ``pip``:

.. code-block:: bash

   $ pip install geneticalgos

**Trivial example**:

We want to find a set of ``X=(x1, x2, x3, x4)`` which maximizes sum(x1, x2, x3, x4),
when each element x is a float from interval (0, 10). Simple answer is: ``x1 = 10, x2 = 10, x3 = 10, x4 = 10``.
First, we define our fitness function (sum) and then gene_intervals for each x.

All other parameters (population size, crossover method, mutation probability, ...) are configured
with default values. However, you can change and tweak them easily - `Advanced usage <https://geneticalgos.readthedocs.io/en/latest/advanced.html>`__.

.. code-block:: python

    import geneticalgos as ga
    import numpy as np

    def custom_fitness_function(chromosome):
      return sum(chromosome)

    gene_intervals = np.array([[0, 10]] * 4)

    # Create genetic algorithms with default values for GA parameters
    # and our fitness function and gene intervals
    ga_model = ga.GeneticAlgo(
        fitness_function=custom_fitness_function,
        gene_intervals=gene_intervals,
    )

    # Start genetic algorithm simulation
    ga_model.simulate()

    # print best solution
    print(ga_model.best_chromosome)

    # print fitness value for best chromosome
    print(ga_model.best_fitness)


When to use GeneticAlgos
________________________

The main goal of the GeneticAlgos is to be `simple` and `powerful`.

* Simple, because it can be used with **basic knowledge of python** (data structures, functions, ...).
* Simple, because it can be used with **basic knowledge of genetic algorithms** (population, chromosome, fitness function, ...).
* Powerful, because **we can tweak many genetic algorithms parameters** very easily and create complex models with the minimum of configuration.

When **not** to use GeneticAlgos
________________________________

Let's be honest, genetic algorithms are very complex algorithms which have a lot of modifications
from a standard scheme.

You should look somewhere else if you need:

* Something other than binary or numerical encoding - like permutation, strings, ...
* Chromosome genes with different encoding within same chromosome - some genes are float numbers and some of them integers.
* An end criterion that is different from a fixed number of generation cycles.

Issues
______

If you encounter any problems, please `file an issue <http://github.com/lkozelnicky/GeneticAlgos/issues>`_
along with a detailed description. Thank you 😃


About GeneticAlgos
__________________

Created by `Lukas Kozelnicky`.

Distributed under the MIT license. See ``LICENSE.txt`` for more information.
