"""This is the 'NNData Class'."""


from enum import Enum
from collections import deque
import numpy as np
import random


class Order(Enum):
    """Enum define training data: 'shuffle' (random) or 'static' (same)."""

    SHUFFLE = 1
    STATIC = 0


class Set(Enum):
    """Enum request for 'training' or 'testing' data."""

    TRAIN = 1
    TEST = 0


class NNData:
    """Create NNData class."""

    Set = Set
    Order = Order

    @staticmethod
    def percentage_limiter(percentage):
        """Accept percentage as float, and return percentage."""
        if percentage < 0:
            return 0
        if percentage > 1:
            return 1
        else:
            return percentage

    def __init__(self, features=None, labels=None, train_factor=.9):
        """Initialize parameters and internal data."""
        self._features = features if features is not None else None
        self._labels = labels if labels is not None else None
        self._train_factor = self.percentage_limiter(train_factor)
        self._train_indices = []
        self._test_indices = []
        self._train_pool = deque()
        self._test_pool = deque()
        self.load_data(features, labels)
        if features and labels is not None:
            self.load_data(features, labels)

    def load_data(self, features=None, labels=None):
        """Use main setter to set feat/lab to None; create lists of lists."""
        if features is None or labels is None:
            self._features = None
            self._labels = None
            self.split_set()
            return
        if len(features) != len(labels):
            self._features = None
            self._labels = None
            self.split_set()
            raise ValueError("Same length not identified.")
        try:
            self._features = np.array(features, dtype=float)
            self._labels = np.array(labels, dtype=float)
        except ValueError:
            self._features = None
            self._labels = None
            self.split_set()
            raise ValueError(f"Construction failed because {len(features)}"
                             f"does not match {len(labels)}.")
        self.split_set()

    def split_set(self, new_train_factor=None):
        """Confirm new_train_factor is None; calculate ex; generate indices."""
        if new_train_factor is not None:
            self._train_factor = self.percentage_limiter(new_train_factor)
        if self._features is None:
            self._train_indices = []
            self._test_indices = []
            return
        if len(self._features) == 0:
            self._train_indices = []
            self._test_indices = []
            return

        num_ex = len(self._features)
        num_train_ex = int(num_ex * self._train_factor)
        all_ind = list(range(num_ex))
        random.shuffle(all_ind)
        self._train_indices = all_ind[:num_train_ex]
        self._test_indices = all_ind[num_train_ex:]

    def prime_data(self, target_set=None, order=None):
        """Load/use deque(s) as indirect indices."""
        if target_set == self.Set.TRAIN or target_set is None:
            self._train_pool.clear()
            self._train_pool.extend(self._train_indices)
            if order == self.Order.SHUFFLE:
                random.shuffle(self._train_pool)
        if target_set == self.Set.TEST or target_set is None:
            self._test_pool.clear()
            self._test_pool.extend(self._test_indices)
            if order == self.Order.SHUFFLE:
                random.shuffle(self._test_pool)

    def get_one_item(self, target_set=None):
        """Return one feat/lab pair as tuple, or None."""
        if target_set == self.Set.TRAIN or target_set is None:
            pool = self._train_pool
        elif target_set == self.Set.TEST:
            pool = self._test_pool
        else:
            pool = self._test_pool
        if not pool:
            return None
        index = pool.popleft()
        return self._features[index], self._labels[index]

    def number_of_samples(self, target_set=None):
        """Return total testing/training ex."""
        if target_set == self.Set.TRAIN:
            return len(self._train_indices)
        if target_set == self.Set.TEST:
            return len(self._test_indices)
        else:
            return len(self._features)

    def pool_is_empty(self, target_set=None):
        """Return T if empty target_set, F if not. If None, use train pool."""
        if target_set == self.Set.TRAIN or target_set is None:
            pool = self._train_pool
        else:
            pool = self._test_pool
        return len(pool) == 0
