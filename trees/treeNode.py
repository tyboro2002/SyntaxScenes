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
        self.value = value
        self.children = []

    def __repr__(self):
        return f"NaryTreeNode(value={self.value}, children={len(self.children)})"

