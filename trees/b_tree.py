import os
from matplotlib import pyplot as plt
from treeNode import BTreeNode
from tree import Tree
from treesSettings import trees_base_path, FPS, FIGSIZE


class BTree(Tree):
    def __init__(self, order, title=None):
        frames_dir = 'frames'
        super().__init__(frames_dir=frames_dir, command=[
            'ffmpeg', '-y', '-framerate', str(FPS), '-i', os.path.join(frames_dir, 'frame_%03d.png'),
            '-r', str(FPS), '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-pix_fmt', 'yuv420p',
            f'{trees_base_path}{self.__class__.__name__.lower()}_tree_animation.mp4'
        ])
        self.order = order
        self.root = None
        self.title = title if title else f'B-Tree of Order {self.order} Visualization'

        # Create directory to save frames
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)

    def insert(self, value):
        if self.root is None:
            self.root = BTreeNode(self.order)
            self.root.values.append(value)
        else:
            if not self._contains(self.root, value):
                result = self._insert(self.root, value)
                if isinstance(result, tuple):
                    # If the root was split, create a new root
                    new_root = BTreeNode(self.order)
                    new_root.values.append(result[1])
                    new_root.children.append(result[0])
                    new_root.children.append(result[2])
                    self.root = new_root
            else:
                return False
        return True

    def _contains(self, node, value):
        """Check if the tree contains a specific value."""
        if value in node.values:
            return True
        if node.is_leaf():
            return False
        index = self._find_child_index(node, value)
        return self._contains(node.children[index], value)

    def _insert(self, node, value):
        if node.is_leaf():
            if value not in node.values:
                node.values.append(value)
                node.values.sort()

                if len(node.values) == self.order:  # Node needs to be split
                    return self._split(node)
            return node
        else:
            # Determine which child to insert into
            index = self._find_child_index(node, value)

            # Recursively insert the value
            result = self._insert(node.children[index], value)
            if isinstance(result, tuple):  # If the child was split
                node.values.insert(index, result[1])
                node.children[index] = result[0]
                node.children.insert(index + 1, result[2])

                if len(node.values) == self.order:  # Node needs to be split
                    return self._split(node)
            return node

    def _find_child_index(self, node, value):
        """Find the index of the child to insert the value into."""
        for i, val in enumerate(node.values):
            if value < val:
                return i
        return len(node.values)

    def _split(self, node):
        self.capture_frame()
        # Split the node and return the left node, middle value, and right node
        mid_index = len(node.values) // 2
        mid_value = node.values[mid_index]

        left_node = BTreeNode(self.order)
        left_node.values = node.values[:mid_index]
        right_node = BTreeNode(self.order)
        right_node.values = node.values[mid_index + 1:]

        if not node.is_leaf():
            left_node.children = node.children[:mid_index + 1]
            right_node.children = node.children[mid_index + 1:]

        return left_node, mid_value, right_node

    def traverse(self, node=None, depth=0):
        if node is None:
            node = self.root
        if node is not None:
            print("  " * depth + str(node.values))
            for child in node.children:
                self.traverse(child, depth + 1)

    def compute_positions(self, node, pos=None, level=0, pos_x=0, x_offset=4):
        """Compute the positions of nodes for visualization."""
        if pos is None:
            pos = {}
        if node is not None:
            pos[str(node.values)] = (pos_x, -level)
            total_children = len(node.children)
            if total_children > 0:
                dx = x_offset / total_children
                current_x = pos_x - (x_offset / 2)
                for child in node.children:
                    current_x += dx
                    pos = self.compute_positions(child, pos, level + 1, current_x, dx)
        return pos

    def animate_insertions(self, values):
        for value in values:
            print(f"Updating frame {self.step} with value {value}")
            if self.insert(value):
                self.capture_frame()

        # Compile images into a video
        self.compile_frames_to_video()

        # Clear the frames directory
        self.clear_frames_directory()

        plt.close()

    def capture_frame(self, file_name=None):
        fig, ax = plt.subplots(figsize=FIGSIZE)

        if self.root is not None:
            positions = self.compute_positions(self.root)

            for node, (x, y) in positions.items():
                # Draw edges to children
                node_obj = next((n for n in self._traverse_nodes() if str(n.values) == node), None)
                if node_obj:
                    for child in node_obj.children:
                        child_x, child_y = positions.get(str(child.values), (x, y))
                        ax.plot([x, child_x], [y, child_y], 'k-', lw=1)

            for node, (x, y) in positions.items():
                node_obj = next((n for n in self._traverse_nodes() if str(n.values) == node), None)
                if node_obj:
                    # Check if the node has too many values
                    if len(node_obj.values) > self.order - 1:
                        facecolor = 'red'  # Node with too many values
                    else:
                        facecolor = 'lightblue'  # Normal node color

                    ax.text(x, y, str(node), ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor=facecolor))

        ax.axis('off')
        ax.set_title(self.title)

        if file_name is None:
            frame_filename = os.path.join(self.frames_dir, f'frame_{self.step:03d}.png')
        else:
            frame_filename = file_name

        plt.savefig(frame_filename, bbox_inches='tight')
        plt.close()
        self.step += 1

    def _traverse_nodes(self, node=None):
        """Traverse all nodes in the BTree, used to find and process nodes."""
        if node is None:
            node = self.root
        yield node
        for child in node.children:
            yield from self._traverse_nodes(child)
