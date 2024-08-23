from binaryTree import BinaryTree
from treeNode import TreeNode


class AVLTree(BinaryTree):
    def height(self, node):
        return 0 if not node else node.height

    def update_height(self, node):
        if node:
            node.height = max(self.height(node.left), self.height(node.right)) + 1

    def get_balance(self, node):
        return 0 if not node else self.height(node.left) - self.height(node.right)

    def right_rotate(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        self.update_height(y)
        self.update_height(x)
        return x

    def left_rotate(self, x):
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        self.update_height(x)
        self.update_height(y)
        return y

    def insert(self, value):
        """Insert a value into the AVL tree."""
        self.root = self._insert(self.root, value)

    def _insert(self, node, value):
        if not node:
            return TreeNode(value)

        if value < node.value:
            node.left = self._insert(node.left, value)
        else:
            node.right = self._insert(node.right, value)

        self.update_height(node)
        balance = self.get_balance(node)

        if balance > 1:
            if value < node.left.value:
                return self.right_rotate(node)
            else:
                node.left = self.left_rotate(node.left)
                return self.right_rotate(node)

        if balance < -1:
            if value > node.right.value:
                return self.left_rotate(node)
            else:
                node.right = self.right_rotate(node.right)
                return self.left_rotate(node)

        return node
