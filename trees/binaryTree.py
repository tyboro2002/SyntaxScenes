import os

import matplotlib.pyplot as plt

from treesSettings import FPS, trees_base_path


class BinaryTree:
    def __init__(self):
        self.root = None
        self.frames_dir = 'frames'
        self.fig, self.ax = plt.subplots()  # Ensure fig and ax are initialized properly
        self.frames = []
        self.step = 0

        # Create directory to save frames
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)

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

    def capture_frame(self):
        pos = self.compute_positions(self.root)
        self.draw_tree(self.ax, pos)

        # Save the current frame as an image
        frame_filename = os.path.join(self.frames_dir, f'frame_{self.step:03d}.png')
        self.fig.savefig(frame_filename)
        print(f"Saved frame {self.step} to {frame_filename}")
        self.step += 1

    def insert(self, value):
        """Insert a value into the tree."""
        raise NotImplementedError("Subclasses should implement this method.")

    def animate_insertions(self, values):
        for value in values:
            print(f"Updating frame {self.step} with value {value}")
            self.insert(value)
            self.capture_frame()
            # self.step += 1

        # Compile images into a video
        self.compile_frames_to_video()

        # Clear the frames directory
        self.clear_frames_directory()

        plt.close(self.fig)

    def compile_frames_to_video(self):
        import subprocess
        print(f"Compiling frames into video with {FPS} FPS...")
        # Command to convert images to video using ffmpeg
        command = [
            'ffmpeg', '-y', '-framerate', str(FPS), '-i', os.path.join(self.frames_dir, 'frame_%03d.png'),
            '-r', str(FPS), '-pix_fmt', 'yuv420p', f'{trees_base_path}{self.__class__.__name__.lower()}_tree_animation.mp4'
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
