class Tree:

    def __init__(self, recursive_arr):
        """
        Initializes a Tree object from a recursive array representation.
        Args:
            recursive_arr (list): A list where the first element is the value of the node,
                                  and the subsequent elements are lists representing the children
                                  of the node in the same recursive format.
        Attributes:
            value: The value of the current node.
            parent: The parent node of the current node (None for the root node).
            children: A list of child nodes.
            val_to_node: A dictionary mapping node values to their corresponding Tree node objects.
        Example:
            Given the input [1, [2, [4], [5]], [3]], the tree structure will be:
                1
               / \
              2   3
             / \
            4   5
        """

        self.value = recursive_arr[0]
        self.parent = None
        self.children = []
        self.val_to_node = {self.value: self}

        for child in recursive_arr[1:]:
            child_node = Tree(child)
            child_node.parent = self
            self.children.append(child_node)
            self.val_to_node = {**self.val_to_node, **child_node.val_to_node}

    def find_lca(self, u, v):
        """
        Find the Lowest Common Ancestor (LCA) of two nodes in a tree.
        The function assumes that the tree nodes have a 'parent' attribute pointing to their parent node,
        and that there is a dictionary 'val_to_node' mapping node values to their corresponding nodes.
        Args:
            u (int): The value of the first node.
            v (int): The value of the second node.
        Returns:
            int: The value of the lowest common ancestor of the two nodes. If one of the nodes is an ancestor
                 of the other, it returns the value of the ancestor node.
        """

        u_order, v_order = [], []

        u_curr = self.val_to_node[u]
        while u_curr:
            u_order.append(u_curr.value)
            u_curr = u_curr.parent

        v_curr = self.val_to_node[v]
        while v_curr:
            v_order.append(v_curr.value)
            v_curr = v_curr.parent

        last_same = None
        for u_ancestor, v_ancestor in zip(reversed(u_order), reversed(v_order)):
            if u_ancestor == v_ancestor:
                last_same = u_ancestor
            else:
                return last_same

        return u if len(u_order) < len(v_order) else v

    def find_distance_to_ancestor(self, value, ancestor_val):
        """
        Finds the distance (number of edges) from a node with the given value to its ancestor with the specified value.
        Args:
            value (any): The value of the node from which the distance is to be calculated.
            ancestor_val (any): The value of the ancestor node to which the distance is to be calculated.
        Returns:
            int: The distance (number of edges) from the node with the given value to its ancestor with the specified value.
        Raises:
            KeyError: If the value or ancestor_val is not found in the tree.
        """

        value_node = self.val_to_node[value]

        dist = 0
        curr = value_node
        while curr.value != ancestor_val:
            curr = curr.parent
            dist += 1

        return dist

    def min_distance_to_lca(self, u, v):
        """
        Calculate the minimum distance from nodes u and v to their lowest common ancestor (LCA).
        Args:
            u: The first node.
            v: The second node.
        Returns:
            int: The minimum distance from either node u or node v to their LCA.
                 Returns -1 if the LCA does not exist.
        """

        lca = self.find_lca(u, v)
        if lca is None:
            return -1

        dist_u = self.find_distance_to_ancestor(u, lca)
        dist_v = self.find_distance_to_ancestor(v, lca)

        return min(dist_u, dist_v)

    def nodes_at_depth(self, depth):
        """
        Finds all nodes at a given depth in the tree.
        Args:
            depth (int): The depth at which to find nodes.
        Returns:
            list: A list of node values at the specified depth.
        """
        result = []
        self._nodes_at_depth(self, depth, 0, result)
        return result

    def _nodes_at_depth(self, node, target_depth, current_depth, result):

        if node is None:
            return
        if current_depth == target_depth:
            result.append(node.value)
            return

        if current_depth < target_depth and len(node.children) == 0:
            result.append(node.value)
            return

        for child in node.children:
            self._nodes_at_depth(child, target_depth, current_depth + 1, result)

    def which_ancestor(self, u, possible_ancestors):

        u_curr = self.val_to_node[u]
        while u_curr:
            if u_curr.value in possible_ancestors:
                return u_curr.value
            u_curr = u_curr.parent

        return None

    def which_ancestor_batched(self, u_batched, possible_ancestors):
        """
        Determines the ancestor for each element in a batch.
        Args:
            u_batched (list): A list of elements for which to find the ancestors.
            possible_ancestors (list): A list of possible ancestors to check against.
        Returns:
            list: A list of ancestors corresponding to each element in u_batched.
        """

        return [self.which_ancestor(u, possible_ancestors) for u in u_batched]
    



import unittest
from tree import Tree


class TestTree(unittest.TestCase):

    def setUp(self):
        self.tree = Tree([1, [2, [4], [5]], [3, [6], [7]]])

    def test_find_lca(self):
        self.assertEqual(self.tree.find_lca(4, 5), 2)
        self.assertEqual(self.tree.find_lca(4, 6), 1)
        self.assertEqual(self.tree.find_lca(3, 4), 1)
        self.assertEqual(self.tree.find_lca(2, 4), 2)

    def test_find_distance_to_ancestor(self):
        self.assertEqual(self.tree.find_distance_to_ancestor(4, 2), 1)
        self.assertEqual(self.tree.find_distance_to_ancestor(4, 1), 2)
        self.assertEqual(self.tree.find_distance_to_ancestor(6, 3), 1)
        self.assertEqual(self.tree.find_distance_to_ancestor(7, 1), 2)

    def test_min_distance_to_lca(self):
        self.assertEqual(self.tree.min_distance_to_lca(4, 5), 1)
        self.assertEqual(self.tree.min_distance_to_lca(4, 6), 2)
        self.assertEqual(self.tree.min_distance_to_lca(3, 4), 1)
        self.assertEqual(self.tree.min_distance_to_lca(2, 4), 0)

    def test_nodes_at_depth(self):
        self.assertEqual(self.tree.nodes_at_depth(0), [1])
        self.assertEqual(self.tree.nodes_at_depth(1), [2, 3])
        self.assertEqual(self.tree.nodes_at_depth(2), [4, 5, 6, 7])
        self.assertEqual(self.tree.nodes_at_depth(1), [2, 3])


if __name__ == "__main__":
    unittest.main()
