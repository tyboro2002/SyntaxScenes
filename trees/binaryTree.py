import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from treesSettings import FPS, trees_base_path


class BinaryTree:
    def __init__(self):
        self.root = None

    def compute_positions(self, root, pos=None, level=0, pos_x=0, x_offset=4):
        """Compute the positions of nodes for visualization."""
        if pos is None:
            pos = {}
        if root is not None:
            pos[root.value] = (pos_x, -level)
            if root.left:
                pos = self.compute_positions(root.left, pos, level + 1, pos_x - x_offset / (2 ** (level + 1)), x_offset)
            if root.right:
                pos = self.compute_positions(root.right, pos, level + 1, pos_x + x_offset / (2 ** (level + 1)),
                                             x_offset)
        return pos

    def draw_tree(self, ax, positions):
        """Draw the binary search tree."""
        ax.clear()
        if self.root is not None:
            for node in positions:
                x, y = positions[node]
                node_obj = self.find_node(self.root, node)
                if node_obj:
                    if node_obj.left:
                        left_x, left_y = positions.get(node_obj.left.value, (x, y))
                        ax.plot([x, left_x], [y, left_y], 'k-', lw=1)
                    if node_obj.right:
                        right_x, right_y = positions.get(node_obj.right.value, (x, y))
                        ax.plot([x, right_x], [y, right_y], 'k-', lw=1)

            for node, (x, y) in positions.items():
                ax.text(x, y, str(node), ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='lightblue'))

        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 1)
        ax.axis('off')
        ax.set_title(f'{self.__class__.__name__} Visualization')

    def find_node(self, root, value):
        """Find a node in the tree."""
        if root is None:
            return None
        if root.value == value:
            return root
        left = self.find_node(root.left, value)
        if left:
            return left
        return self.find_node(root.right, value)

    def insert(self, value):
        """Insert a value into the tree."""
        raise NotImplementedError("Subclasses should implement this method.")

    def animate_insertions(self, values):
        """Animate the insertion of values into the binary search tree."""
        fig, ax = plt.subplots()
        done_frames = set()
        frames = []

        def update(frame):
            nonlocal done_frames
            if frame in done_frames:
                return
            done_frames.add(frame)
            print(f"new frame with {frame} {values[frame]}")
            self.insert(values[frame])
            pos = self.compute_positions(self.root)
            self.draw_tree(ax, pos)
            fig.canvas.draw()

            # Convert canvas to image
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(img)
            return []

        ani = animation.FuncAnimation(fig, update, frames=len(values), repeat=False, interval=1000)
        ani.save(f'{trees_base_path}{self.__class__.__name__.lower()}_tree_animation.mp4', writer='ffmpeg', fps=FPS)
