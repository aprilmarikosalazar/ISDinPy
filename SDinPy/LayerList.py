"""This is the 'LayerList Class'."""


from DoublyLinkedList import DoublyLinkedList
from Neurode import Neurode


class LayerList(DoublyLinkedList):
    """Create a LayerList class; extend DLLNode to LayerList."""

    def __init__(self, inputs: int, outputs: int, neurode_type: type(Neurode)):
        """Set constructor for 3 arguments and call to super."""
        super().__init__()
        self._neurode_type = neurode_type
        if inputs < 1 or outputs < 1:
            raise ValueError
        input_layer = [neurode_type() for _ in range(inputs)]
        output_layer = [neurode_type() for _ in range(outputs)]
        self.add_to_head(input_layer)
        self.add_after_current(output_layer)
        self._link_to_next()

    def add_layer(self, num_nodes: int):
        """Create hidden after current layer;
        link with UP and DOWN -STREAM neurode neighbors."""
        if self._curr == self._tail:
            raise IndexError("Error: Current layer is the output layer.")
        if num_nodes < 0:
            raise ValueError
        hidden_layer = [self._neurode_type() for _ in range(num_nodes)]
        self.add_after_current(hidden_layer)
        self._link_to_next()
        self.move_forward()
        self._link_to_next()
        self.move_backward()

    def remove_layer(self):
        """Remove layer after current; link remaining layer neurodes."""
        if self._curr == self._tail or self._curr.next == self._tail:
            raise IndexError("Error: Cannot remove the output layer.")
        self.remove_after_current()
        self._link_to_next()

    @property
    def input_nodes(self):
        """Return list of input layer neurodes."""
        return self._head.data

    @property
    def output_nodes(self):
        """Return list of output layer neurodes."""
        return self._tail.data

    def _link_to_next(self):
        """Add helper method to connect current and next nodes."""
        for node in self._curr.data:
            node.reset_neighbors(self._curr.next.data,
                                 self._neurode_type.Side.DOWNSTREAM)
        for node in self._curr.next.data:
            node.reset_neighbors(self._curr.data,
                                 self._neurode_type.Side.UPSTREAM)
