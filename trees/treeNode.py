class TreeNode:
    def __init__(self, value, left=None, right=None, color='black'):
        self.value = value
        self.left = left
        self.right = right
        self.height = 1  # Used for AVL Tree
        self.parent = None  # Used for red black
        self.color = color  # Add color attribute # Used for red black

    def __str__(self):
        left = self.left.value if self.left else "None"
        right = self.right.value if self.right else "None"
        return f"node: {self.value} left: {left} right: {right}"


class NaryTreeNode:
    def __init__(self, value=None):
        self.values = [value] if value is not None else []
        self.children = []

    def __repr__(self):
        return f"NaryTreeNode(value={self.values}, children={len(self.children)})"


class BTreeNode:
    def __init__(self, order):
        self.order = order  # Maximum number of children is order
        self.values = []  # Store keys as a list
        self.children = []  # Children will also be a list

    def is_leaf(self):
        return len(self.children) == 0

    def __repr__(self):
        return f"BTreeNode(values={self.values}, children={len(self.children)})"
