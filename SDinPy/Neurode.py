"""This is the 'Neurode Class'."""


from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
import random


class MultiLinkNode(ABC):
    """Create abstract base (inner) class 'MultiLinkNode';
     include methods for future FFBPNeurode class."""

    class Side(Enum):
        """Establish Side inner class for only MultiLinkNode instances;
        identify relationships between neurodes."""

        UPSTREAM = 1
        DOWNSTREAM = 0

    def __init__(self):
        """Initialize 3 instance attributes."""
        self._reporting_nodes = {self.Side.UPSTREAM: 0,
                                 self.Side.DOWNSTREAM: 0}
        self._reference_value = {self.Side.UPSTREAM: 0,
                                 self.Side.DOWNSTREAM: 0}
        self._neighbors = {self.Side.UPSTREAM: [],
                           self.Side.DOWNSTREAM: []}

    def __str__(self):
        """Return string representation of node in context."""
        up_id = [node.__str__() for node in
                 self._neighbors[self.Side.UPSTREAM]]
        down_id = [node.__str__() for node in
                   self._neighbors[self.Side.DOWNSTREAM]]
        print(f"Node ID = {id(self)}, "
              f"UPSTREAM ID = {up_id}, "
              f"DOWNSTREAM ID = {down_id}")

    @abstractmethod
    def _process_new_neighbor(self, node: MultiLinkNode, side: Side):
        """Establish abstract method to implement in Neurode class."""
        pass

    def reset_neighbors(self, nodes: list, side: Side):
        """Clear neighbors value and populate new upstream/downstream entry."""
        self._neighbors[side] = list(nodes)
        self._reference_value[side] = 2 ** len(nodes) - 1
        for node in nodes:
            self._process_new_neighbor(node, side)


class Neurode(MultiLinkNode):
    """Create Neurode class to implement MultiLinkNode."""

    _learning_rate = .05

    @property
    def learning_rate(self):
        """Run 'learning_rate' getter."""
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, new_lr):
        """Run 'learning_rate' setter."""
        self._learning_rate = new_lr

    @property
    def value(self):
        """Run 'value' getter."""
        return self._value

    def __init__(self):
        """Initialize attributes and call MultiLinkNode class constructor."""
        super().__init__()
        self._value = 0
        self._weights = {}

    def _process_new_neighbor(self, node: Neurode, side: Side):
        """Generate weight of added upstream neighbor,
        then train up/down."""
        if side == self.Side.UPSTREAM:
            self._weights[node] = random.random()

    def _check_in(self, node: Neurode, side: Side):
        """Report neighbor node index,
        and validate if all neighbor nodes reported."""
        index = self._neighbors[side].index(node)
        self._reporting_nodes[side] |= 2 ** index
        if self._reporting_nodes[side] == self._reference_value[side]:
            self._reporting_nodes[side] = 0
            return True
        return False

    def get_weight(self, node: Neurode):
        """Use current node to return upstream node weight."""
        return self._weights.get(node, 0)
