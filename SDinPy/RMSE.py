"""This is the 'RMSE Class':
choose method to calculate distance using constructor injection."""


from abc import ABC, abstractmethod
from math import sqrt


class RMSE(ABC):
    """Create an interface called RMSE to support child classes."""

    def __init__(self):
        """Create constructor."""
        self.predict_values = []
        self.expect_values = []

    @property
    def error(self):
        """Calculate and return total RMSE error."""
        if len(self.predict_values) != len(self.expect_values):
            raise ValueError("Error: Value lengths are not equal.")
        if not self.predict_values:
            return 0
        error_tot = sum(self.distance(pred, exp) ** 2 for pred, exp
                        in zip(self.predict_values, self.expect_values))
        return sqrt(error_tot / len(self.predict_values))

    @staticmethod
    @abstractmethod
    def distance(vector_one, vector_two):
        """Implement method in child class."""
        pass

    def __add__(self, other):
        """Create new object to allow user to add predicted/expected values."""
        if isinstance(other, RMSE):
            combine = self.__class__()
            combine.predict_values = self.predict_values + other.predict_values
            combine.expect_values = self.expect_values + other.expect_values
            return combine
        elif isinstance(other, tuple):
            new = self.__class__()
            new.predict_values = self.predict_values + [other[0]]
            new.expect_values = self.expect_values + [other[1]]
            return new
        else:
            raise TypeError

    def __iadd__(self, other):
        """Support add method to avoid creating new objects
        and conserve time for NN training."""
        if isinstance(other, RMSE):
            self.predict_values += other.predict_values
            self.expect_values += other.expect_values
            return self
        elif isinstance(other, tuple):
            self.predict_values.append(other[0])
            self.expect_values.append(other[1])
            return self
        else:
            raise TypeError

    def reset(self):
        """Clear all internal data."""
        self.predict_values = []
        self.expect_values = []


class Euclidean(RMSE):
    """Return distance between 2 points through Euclidean distance formula."""

    @staticmethod
    def distance(vector_one, vector_two):
        return sqrt(sum((a - b) ** 2 for a, b in zip(vector_one, vector_two)))


class Taxicab(RMSE):
    """Return distance between 2 points through Taxicab distance formula."""

    @staticmethod
    def distance(vector_one, vector_two):
        return sum(abs(a - b) for a, b in zip(vector_one, vector_two))
