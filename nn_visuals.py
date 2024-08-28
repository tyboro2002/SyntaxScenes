import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def generate_yin_yang_data(n_samples=1000):
    """Generate a synthetic Yin-Yang dataset."""
    np.random.seed(41)

    # Create a mesh grid
    x = np.linspace(-1, 1, int(np.sqrt(n_samples)))
    y = np.linspace(-1, 1, int(np.sqrt(n_samples)))
    x, y = np.meshgrid(x, y)

    # Flatten the grid
    x = x.flatten()
    y = y.flatten()

    # Generate labels for Yin-Yang pattern
    labels = ((x ** 2 + y ** 2 < 0.5 ** 2) & (np.sqrt(x ** 2 + y ** 2) < 0.25)).astype(int)

    # Add some random noise
    x += np.random.randn(len(x)) * 0.05
    y += np.random.randn(len(y)) * 0.05

    # Combine x and y into feature array
    X = np.column_stack((x, y))

    return X, labels


class NeuralNetworkVisualizer:
    def __init__(self, X, y, model, num_steps, frames_dir='frames', fps=2):
        self.X = X
        self.y = y
        self.model = model
        self.num_steps = num_steps
        self.frames_dir = frames_dir
        self.fps = fps  # Frames per second

        # Create directory to save frames
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.axis('off')  # Hide the axis
        self.frame_count = 0

    def plot_decision_boundary(self):
        # Create a grid of points
        h = .02  # Step size in the mesh
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predict on each point in the grid
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary and margins
        self.ax.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
        self.ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, edgecolor='k', cmap='viridis')

    def capture_frame(self, step):
        self.ax.clear()
        self.ax.axis('off')  # Hide the axis
        self.plot_decision_boundary()
        accuracy = accuracy_score(self.y, self.model.predict(self.X))
        self.ax.set_title(f"Step {step}: Accuracy = {accuracy:.2f}")

        # Save the current frame as an image
        frame_filename = os.path.join(self.frames_dir, f'frame_{step:03d}.png')
        self.fig.savefig(frame_filename)
        print(f"Saved frame {step} to {frame_filename}")

    def train_and_visualize(self):
        for step in range(1, self.num_steps + 1):
            print(f"Training step {step}")
            self.model.partial_fit(self.X, self.y, classes=np.unique(self.y))
            self.capture_frame(step)

        # Compile images into a video
        self.compile_frames_to_video()

        # Clear the frames directory
        self.clear_frames_directory()

        plt.close(self.fig)

    def compile_frames_to_video(self):
        import subprocess
        print(self.fps)
        # Command to convert images to video using ffmpeg
        command = [
            'ffmpeg', '-y', '-framerate', str(self.fps), '-i', os.path.join(self.frames_dir, 'frame_%03d.png'),
            '-r', '2', '-pix_fmt', 'yuv420p', 'neural_network_training.mp4'
        ]

        subprocess.run(command, check=True)
        print("Video saved as neural_network_training.mp4")

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


# Example usage
# Generate synthetic data
# X, y = make_classification(
#     n_samples=200,
#     n_features=2,
#     n_informative=2,
#     n_redundant=0,
#     n_repeated=0,
#     n_classes=2,
#     n_clusters_per_class=1
# )
# Generate synthetic data
X, y = generate_yin_yang_data(n_samples=1000)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = MLPClassifier(
    hidden_layer_sizes=(50, 30),  # Increase the number of neurons and layers
    max_iter=1,
    warm_start=True,
    solver='adam')
visualizer = NeuralNetworkVisualizer(X_scaled, y, model, num_steps=250, fps=25)
visualizer.train_and_visualize()
