from binaryTree import BinaryTree
from treeNode import TreeNode

class RedBlackTree(BinaryTree):
    def __init__(self):
        super().__init__()
        self.NIL = TreeNode(value=None, color='black', left=None, right=None)
        self.root = self.NIL  # Initialize root to NIL

    def insert(self, value):
        if self.root == self.NIL:
            self.root = TreeNode(value=value, left=self.NIL, right=self.NIL, color='black')
        else:
            new_node = self._insert(self.root, value)
            new_node.color = 'red'
            self.fix_insert(new_node)
        self.root.color = 'black'  # Ensure the root is always black

    def _insert(self, node, value) -> TreeNode:
        if node is None or node.value is None:
            return TreeNode(value=value, left=self.NIL, right=self.NIL, color='red')

        if value < node.value:
            if node.left == self.NIL:
                node.left = TreeNode(value=value, left=self.NIL, right=self.NIL, color='red')
                node.left.parent = node
                return node.left
            else:
                return self._insert(node.left, value)
        else:
            if node.right == self.NIL:
                node.right = TreeNode(value=value, left=self.NIL, right=self.NIL, color='red')
                node.right.parent = node
                return node.right
            else:
                return self._insert(node.right, value)

    def fix_insert(self, node):
        while node != self.root and node.parent and node.parent.color == 'red':
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle and uncle.color == 'red':
                    node.parent.color = 'black'
                    uncle.color = 'black'
                    node.parent.parent.color = 'red'
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        node = node.parent
                        self.left_rotate(node)
                    node.parent.color = 'black'
                    node.parent.parent.color = 'red'
                    self.right_rotate(node.parent.parent)
            else:
                uncle = node.parent.parent.left
                if uncle and uncle.color == 'red':
                    node.parent.color = 'black'
                    uncle.color = 'black'
                    node.parent.parent.color = 'red'
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self.right_rotate(node)
                    node.parent.color = 'black'
                    node.parent.parent.color = 'red'
                    self.left_rotate(node.parent.parent)

        self.root.color = 'black'

    def left_rotate(self, x) -> TreeNode:
        # Capture the state before rotation
        self.capture_frame()

        y = x.right
        x.right = y.left
        if y.left != self.NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        return y

    def right_rotate(self, y) -> TreeNode:
        # Capture the state before rotation
        self.capture_frame()

        x = y.left
        y.left = x.right
        if x.right != self.NIL:
            x.right.parent = y
        x.parent = y.parent
        if y.parent is None:
            self.root = x
        elif y == y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
        x.right = y
        y.parent = x
        return x

    def draw_tree(self, ax, positions):
        """Draw the Red-Black Tree with color-coding."""
        ax.clear()
        if self.root is not None:
            for node in positions:
                if node is not None:
                    x, y = positions[node]
                    node_obj = self.find_node(self.root, node)
                    if node_obj:
                        color = 'red' if node_obj.color == 'red' else 'black'
                        if node_obj.left and node_obj.left != self.NIL and node_obj.left.value is not None:
                            left_x, left_y = positions.get(node_obj.left.value, (x, y))
                            ax.plot([x, left_x], [y, left_y], color='k', lw=1)
                        if node_obj.right and node_obj.right != self.NIL and node_obj.right.value is not None:
                            right_x, right_y = positions.get(node_obj.right.value, (x, y))
                            ax.plot([x, right_x], [y, right_y], color='k', lw=1)

            for node, (x, y) in positions.items():
                if node is not None:
                    node_obj = self.find_node(self.root, node)
                    color = 'red' if node_obj.color == 'red' else 'black'
                    ax.text(x, y, str(node), ha='center', va='center', color='white',
                            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor=color))

        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 1)
        ax.axis('off')
        ax.set_title(f'{self.__class__.__name__} Visualization')

