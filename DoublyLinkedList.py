"""This is the 'DoublyLinkedList Class'."""


class DLLNode:
    """Create a DLLNode class."""

    def __init__(self, data):
        """Implement attributes '.next' and '.prev'."""
        self.data = data
        self.next = None
        self.prev = None


class DoublyLinkedList:
    """Implement Doubly-Linked List ADT using public methods."""

    def __init__(self):
        """Implement attributes '._head', '._curr', and '._tail'."""
        self._head = None
        self._curr = None
        self._tail = None

    @property
    def curr_data(self):
        """Return the data at the current position."""
        if not self._curr:
            raise IndexError
        return self._curr.data

    def add_to_head(self, data):
        """Add a node to the head of the list."""
        new_node = DLLNode(data)
        new_node.next = self._head
        if self._head:
            self._head.prev = new_node
        self._head = new_node
        if not self._tail:
            self._tail = new_node
        self.reset_to_head()

    def add_after_current(self, data):
        """Add a node after the current node, and return the data."""
        if not self._curr:
            raise IndexError
        new_node = DLLNode(data)
        new_node.next = self._curr.next
        new_node.prev = self._curr
        if self._curr.next:
            self._curr.next.prev = new_node
        self._curr.next = new_node
        if self._curr == self._tail:
            self._tail = new_node

    def remove_from_head(self):
        """Remove a node from the head of the list."""
        if not self._head:
            raise IndexError
        return_value = self._head.data
        self._head = self._head.next
        if self._head:
            self._head.prev = None
        if not self._head:
            self._tail = None
        self.reset_to_head()
        return return_value

    def remove_after_current(self):
        """Remove a node after the current node, and return the data."""
        if not self._curr or not self._curr.next:
            raise IndexError
        return_value = self._curr.next.data
        self._curr.next = self._curr.next.next
        if self._curr.next:
            self._curr.next.prev = self._curr
        if not self._curr.next:
            self._tail = self._curr
        return return_value

    def reset_to_head(self):
        """Reset the current pointer to the head."""
        self._curr = self._head

    def reset_to_tail(self):
        """Reset the current pointer to the tail."""
        self._curr = self._tail

    def move_forward(self):
        """Move forward through the list."""
        if not self._curr or not self._curr.next:
            raise IndexError
        self._curr = self._curr.next

    def move_backward(self):
        """Move backward through the list."""
        if not self._curr or not self._curr.prev:
            raise IndexError
        self._curr = self._curr.prev

    def find(self, data):
        """Find an item in the list, and return the data."""
        temp_curr = self._head
        while temp_curr:
            if temp_curr.data == data:
                return temp_curr.data
            temp_curr = temp_curr.next
        raise IndexError

    def remove(self, data):
        """Find a node, then remove."""
        if not self._head:
            raise IndexError
        if self._head.data == data:
            return self.remove_from_head()
        temp_curr = self._head
        while temp_curr.next:
            if temp_curr.next.data == data:
                return_value = temp_curr.next.data
                temp_curr.next = temp_curr.next.next
                if temp_curr.next:
                    temp_curr.next.prev = temp_curr
                else:
                    self._tail = temp_curr
                self.reset_to_head()
                return return_value
            temp_curr = temp_curr.next
        raise IndexError
