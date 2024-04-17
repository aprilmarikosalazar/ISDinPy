"""This is the 'BPNeurode Class'."""


from __future__ import annotations
from Neurode import Neurode


class BPNeurode(Neurode):
    """Create BPNeurode class."""

    def __init__(self):
        """Call to super; initialize BPNeurode prediction error to 0."""
        super().__init__()
        self._delta = 0

    @property
    def delta(self):
        """Run 'delta' getter."""
        return self._delta

    @staticmethod
    def _sigmoid_derivative(value):
        """Calculate derivative of f(x) * (1 - f(x))."""
        return value * (1 - value)

    def _calculate_delta(self, expected_value: float = None):
        """Check if output layer neurode; then save result."""
        if expected_value is not None:
            error = expected_value - self.value
            self._delta = error * self._sigmoid_derivative(self.value)
        else:
            error = 0
            for neurode in self._neighbors[Neurode.Side.DOWNSTREAM]:
                error += neurode.get_weight(self) * neurode.delta
            self._delta = error * self._sigmoid_derivative(self.value)

    def data_ready_downstream(self, from_node: BPNeurode):
        """Register node has data; then collect and move it to layer up."""
        if self._check_in(from_node, Neurode.Side.DOWNSTREAM):
            self._calculate_delta()
            self._fire_upstream()
            self._update_weights()

    def set_expected(self, expected_value: float):
        """Directly set output layer neurode expected value."""
        self._calculate_delta(expected_value)
        self._fire_upstream()

    def adjust_weights(self, node: Neurode, adjustment: float):
        """Use attn-seeking upstream node to call method."""
        self._weights[node] += adjustment

    def _update_weights(self):
        """Adjust weight/importance of node's data."""
        for node in self._neighbors[Neurode.Side.DOWNSTREAM]:
            adjustment = node.learning_rate * node.delta * self.value
            node.adjust_weights(self, adjustment)

    def _fire_upstream(self):
        """Call method for each upstream node's neighbors."""
        for node in self._neighbors[Neurode.Side.UPSTREAM]:
            node.data_ready_downstream(self)
