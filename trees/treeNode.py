class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        self.height = 1  # Used for AVL Tree

    def __str__(self):
        left = self.left.value if self.left else "None"
        right = self.right.value if self.right else "None"
        return f"node: {self.value} left: {left} right: {right}"
