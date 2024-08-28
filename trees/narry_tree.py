import networkx as nx
from matplotlib import pyplot as plt

from treeNode import NaryTreeNode


class NaryTree:
    def __init__(self, n):
        self.n = n  # Maximum number of children per node
        self.root = None

    def insert(self, value):
        new_node = NaryTreeNode(value)
        if self.root is None:
            self.root = new_node
        else:
            self._insert_in_subtree(self.root, new_node)

    def _insert_in_subtree(self, node, new_node):
        if len(node.children) < self.n:
            node.children.append(new_node)
        else:
            # Recursively try to insert in the child nodes
            for child in node.children:
                if len(child.children) < self.n:
                    self._insert_in_subtree(child, new_node)
                    return
            # If all child nodes are full, insert in the first child's subtree
            self._insert_in_subtree(node.children[0], new_node)

    def traverse(self, node=None, depth=0):
        if node is None:
            node = self.root
        if node is not None:
            print("  " * depth + str(node.value))
            for child in node.children:
                self.traverse(child, depth + 1)

    def compute_positions(self, node, pos=None, level=0, pos_x=0, x_offset=4):
        """Compute the positions of nodes for visualization."""
        if pos is None:
            pos = {}
        if node is not None:
            pos[node.value] = (pos_x, -level)
            total_children = len(node.children)
            if total_children > 0:
                dx = x_offset / total_children
                current_x = pos_x - (x_offset / 2)
                for child in node.children:
                    current_x += dx
                    pos = self.compute_positions(child, pos, level + 1, current_x, dx)
        return pos

    def capture_frame(self, file_name='tree.png'):
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.DiGraph()
        pos = {}

        def add_edges(node):
            if node:
                G.add_node(node.value)
                for child in node.children:
                    G.add_edge(node.value, child.value)
                    add_edges(child)

        add_edges(self.root)
        pos = self.compute_positions(self.root)

        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold',
                font_color='black', arrows=False, edge_color='gray')
        plt.title('N-ary Tree Visualization')
        plt.savefig(file_name)
        plt.close()