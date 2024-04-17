"""This is the 'FFNeurode Class'."""


from __future__ import annotations
from Neurode import Neurode
from numpy import exp


class FFNeurode(Neurode):
    """Create FFNeurode class."""

    def __init__(self):
        """Use the constructor to call to super."""
        super().__init__()
        self.bias = 0

    @staticmethod
    def _sigmoid(value: float) -> float:
        """Use exponential function to get sigmoid as a float."""
        return 1 / (1 + exp(-value))

    def _calculate_value(self):
        """Calculate weight sum of UPSTREAM nodes,
        pass through _sigmoid, and store value."""
        sum_weight = self.bias
        for node, weight in self._weights.items():
            sum_weight += weight * node.value
        self._value = self._sigmoid(sum_weight)

    def _fire_downstream(self):
        """Call data_ready_upstream method on downstream neighbors."""
        for node in self._neighbors[self.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)

    def data_ready_upstream(self, node: Neurode):
        """Pass neurode object reference to self onto other neurode."""
        side = self.Side.UPSTREAM
        if self._check_in(node, side):
            self._calculate_value()
            self._fire_downstream()

    def set_input(self, input_value: float):
        """Set input layer neurode value;
        call data_ready_upstream on downstream neighbors."""
        self._value = input_value
        self._fire_downstream()
