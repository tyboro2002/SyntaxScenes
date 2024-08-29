import os

import networkx as nx
from matplotlib import pyplot as plt

from treeNode import NaryTreeNode
from treesSettings import trees_base_path, FPS


class NaryTree:
    def __init__(self):
        self.root = None
        self.step = 0
        self.frames_dir = 'frames'

        # Create directory to save frames
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)

    def insert(self, value):
        if self.root is None:
            self.root = NaryTreeNode(value)
        else:
            # Insert the value and check if the root needs to be split
            result = self._insert(self.root, value)
            if isinstance(result, tuple):
                # If the root was split, create a new root
                new_root = NaryTreeNode(result[2])
                new_root.children.append(result[0])
                new_root.children.append(result[1])
                self.root = new_root

    def _insert(self, node, value):
        # If the node is a leaf, insert the value directly
        if len(node.children) == 0:
            node.values.append(value)
            node.values.sort()

            if len(node.values) == 3:  # Node needs to be split
                return self._split(node)
            return node
        else:
            # Determine which child to insert into
            if value < node.values[0]:
                index = 0
            elif len(node.values) == 1 or value < node.values[1]:
                index = 1
            else:
                index = 2

            # Recursively insert the value
            result = self._insert(node.children[index], value)
            if isinstance(result, tuple):  # If the child was split
                node.children[index] = result[0]
                node.children.insert(index + 1, result[1])
                node.values.insert(index, result[2])

                if len(node.values) == 3:  # Node needs to be split
                    return self._split(node)
            return node

    def _split(self, node):
        # Split the node into two nodes and push the middle value up
        left_node = NaryTreeNode(node.values[0])
        right_node = NaryTreeNode(node.values[2])

        if len(node.children) > 0:
            left_node.children = node.children[:2]
            right_node.children = node.children[2:]

        return left_node, right_node, node.values[1]

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

    def capture_frame(self, file_name=None):
        G = nx.DiGraph()
        pos = {}

        def add_edges(node):
            if node:
                G.add_node(str(node.values))
                for child in node.children:
                    G.add_edge(str(node.values), str(child.values))
                    add_edges(child)

        add_edges(self.root)
        pos = self.compute_positions(self.root)

        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold',
                font_color='black', arrows=False, edge_color='gray')
        plt.title('2-3 Tree Visualization')
        if file_name is None:
            frame_filename = os.path.join(self.frames_dir, f'frame_{self.step:03d}.png')
        else:
            frame_filename = file_name
        plt.savefig(frame_filename)
        plt.close()

    def animate_insertions(self, values):
        for value in values:
            print(f"Updating frame {self.step} with value {value}")
            self.insert(value)
            self.capture_frame()
            self.step += 1

        # Compile images into a video
        self.compile_frames_to_video()

        # Clear the frames directory
        self.clear_frames_directory()

        plt.close()

    def compile_frames_to_video(self):
        import subprocess
        print(f"Compiling frames into video with {FPS} FPS...")
        # Command to convert images to video using ffmpeg
        command = [
            'ffmpeg', '-y', '-framerate', str(FPS), '-i', os.path.join(self.frames_dir, 'frame_%03d.png'),
            '-r', str(FPS), '-pix_fmt', 'yuv420p',
            f'{trees_base_path}{self.__class__.__name__.lower()}_tree_animation.mp4'
        ]

        subprocess.run(command, check=True)
        print(f"Video saved as {trees_base_path}{self.__class__.__name__.lower()}_tree_animation.mp4")

    def clear_frames_directory(self):
        # Remove all files in the frames directory
        for filename in os.listdir(self.frames_dir):
            file_path = os.path.join(self.frames_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted file {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

        # Optionally, remove the directory itself
        os.rmdir(self.frames_dir)
        print(f"Cleared and removed frames directory: {self.frames_dir}")
