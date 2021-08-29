class FitnessFunctionNotCallable(TypeError):
    """Fitness function is not callable."""

    pass


class FitnessFunctionReturnsNone(ValueError):
    """Fitness function returns None."""

    pass


class InvalidGeneIntervalsShape(ValueError):
    """Invalid shape for gene intervals."""

    pass


class InvalidNumberOfIterations(ValueError):
    """Invalid number of iterations for simulation."""

    pass


class InvalidObjectiveGoal(ValueError):
    """Invalid objective goal for fitness function."""

    pass


class InvalidChromosomeType(TypeError):
    """Invalid chromosome type."""

    pass
