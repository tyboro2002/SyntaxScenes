import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

frames_dir = 'frames/'
output_dir = 'nn_visuals/'
FPS = 25
target_accuracy = 0.99
max_itter = 100
selected_pattern = 2
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Step 1: Define the patterns and labels
patterns = np.array([
    [0, 0, 0, 0],  # All 0s
    [1, 1, 1, 1],  # All 1s
    [1, 0, 0, 1],  # Diagonal TL-BR
    [0, 1, 1, 0],  # Diagonal TR-BL
    [1, 0, 1, 0],  # Cross pattern
    [0, 1, 0, 1],  # Inverse cross
])

labels = np.array([
    0,  # All same
    0,  # All same
    1,  # Diagonal
    1,  # Diagonal
    2,  # Cross
    2,  # Cross
])

def compile_frames_to_video():
    import subprocess
    print(f"Compiling frames into video with {FPS} FPS...")
    # Command to convert images to video using ffmpeg
    command = [
        'ffmpeg', '-y', '-framerate', str(FPS), '-i', os.path.join(frames_dir, 'neural_network_iteration_%05d.png'),
        '-r', str(FPS), '-pix_fmt', 'yuv420p',
        f'{output_dir}nn_animation.mp4'
    ]

    subprocess.run(command, check=True)
    print(f"Video saved as {output_dir}_nn_animation.mp4")


def clear_frames_directory():
    # Remove all files in the frames directory
    for filename in os.listdir(frames_dir):
        file_path = os.path.join(frames_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted file {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    # Optionally, remove the directory itself
    os.rmdir(frames_dir)
    print(f"Cleared and removed frames directory: {frames_dir}")

# Step 2: Create the neural network (with warm_start=True to allow incremental learning)
nn = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1, activation='relu', solver='adam', random_state=None,
                   warm_start=True)

# Step 3: Define a target accuracy threshold
current_accuracy = 0


# Function to compute activations and draw the neural network
def draw_neural_net(ax, left, right, bottom, top, layer_sizes, coefs_, intercepts_, activations, show_text=True):
    """
    Draw a neural network cartoon using matplotlib.
    left, right, bottom, top: boundaries for the network
    layer_sizes: list containing the number of neurons in each layer
    coefs_: list containing weight matrices for each layer
    intercepts_: list containing bias vectors for each layer
    activations: list containing activations at each layer
    show_text: boolean to control whether to show text values or not
    """
    # Adjust the vertical spacing to make the nodes more compact
    v_spacing = (top - bottom) / float(max(layer_sizes))

    # Adjust horizontal spacing slightly to make nodes closer together without stretching
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Nodes
    for i, n in enumerate(layer_sizes):
        layer_top = v_spacing * (n - 1) / 2. + (top + bottom) / 2.
        for m in range(n):
            # Draw the circle representing a neuron, ensure the radius doesn't stretch
            circle_radius = min(v_spacing, h_spacing) / 4.

            # print(input_layer_activations[selected_pattern])
            if i == 0:
                # Input layer
                neuron_color = 'gray' if input_layer_activations[selected_pattern][m] == 1.00 else 'w'
            elif i == len(layer_sizes) - 1:
                # Output layer
                neuron_color = 'gray' if output_layer_outputs[selected_pattern][m] == output_layer_outputs[selected_pattern].max() else 'w'
            else:
                # Hidden layer
                neuron_color = 'w'

            circle = plt.Circle((i * h_spacing + left, layer_top - m * v_spacing), circle_radius,
                                color=neuron_color, ec='k', zorder=2)
            ax.add_artist(circle)

            if show_text:
                # Extract the activation value, ensuring it's a scalar
                activation_value = activations[i][m] if i < len(activations) else 0
                if isinstance(activation_value, np.ndarray):
                    if activation_value.size == 1:
                        activation_value = activation_value.item()  # Convert to scalar if it has a single element
                    else:
                        activation_value = activation_value[0]  # Use the first element if it's a multi-element array

                # Biases for current layer
                if i > 0:
                    bias = intercepts_[i - 1][m]
                    ax.text(i * h_spacing + left, layer_top - m * v_spacing + 0.05, f'b={bias:.2f}', fontsize=8,
                            ha='center', zorder=3)

                # Now activation_value should be a scalar
                # Label nodes with their index and activation values
                if i == 0:
                    ax.text(i * h_spacing + left - 0.075, layer_top - m * v_spacing, f'I{m + 1}', fontsize=12, zorder=3)
                    ax.text(i * h_spacing + left, layer_top - m * v_spacing + 0.02, f'{activation_value:.2f}',
                            fontsize=10, color='red', ha='center', zorder=3)
                elif i == len(layer_sizes) - 1:
                    ax.text(i * h_spacing + left + 0.07, layer_top - m * v_spacing, f'O{m + 1}', fontsize=12, zorder=3)
                    ax.text(i * h_spacing + left, layer_top - m * v_spacing - 0.02, f'{activation_value:.2f}',
                            fontsize=10, color='green', ha='center', zorder=3)

                else:
                    ax.text(i * h_spacing + left - 0.075, layer_top - m * v_spacing, f'H{m + 1}', fontsize=12, zorder=3)
                    ax.text(i * h_spacing + left, layer_top - m * v_spacing + 0.02, f'{activation_value:.2f}',
                            fontsize=10, color='red', ha='center', zorder=3)
                    ax.text(i * h_spacing + left, layer_top - m * v_spacing - 0.02,
                            f'{np.maximum(0, activation_value):.2f}', fontsize=10, color='green', ha='center', zorder=3)

    # Edges with weight annotations at 1/4 of the connection
    for i in range(len(layer_sizes) - 1):
        layer_top_a = v_spacing * (layer_sizes[i] - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_sizes[i + 1] - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_sizes[i]):
            for n in range(layer_sizes[i + 1]):
                weight = coefs_[i][m, n]
                ax.plot([i * h_spacing + left, (i + 1) * h_spacing + left],
                        [layer_top_a - m * v_spacing, layer_top_b - n * v_spacing], 'k', lw=abs(weight) * 5, zorder=1,
                        alpha=0.5)

                # Add weight annotation at 1/4 of the connection
                x_pos = i * h_spacing + left + (1 / 4) * (h_spacing)
                y_pos = (3 / 4) * (layer_top_a - m * v_spacing) + (1 / 4) * (layer_top_b - n * v_spacing)
                if show_text:
                    ax.text(x_pos, y_pos, f'{weight:.2f}', fontsize=10, ha='center', va='center', color='red', zorder=4)


# Training loop
iteration = 0
while current_accuracy < target_accuracy and iteration < max_itter:
    iteration += 1

    # Incremental training with max_iter=1 for each step
    nn.partial_fit(patterns, labels, classes=np.unique(labels))

    # Step 4: Get the weights and biases
    weights_input_hidden = nn.coefs_[0]
    weights_hidden_output = nn.coefs_[1]
    biases_input_hidden = nn.intercepts_[0]
    biases_hidden_output = nn.intercepts_[1]

    # Step 5: Get the activations of the neurons
    input_layer_activations = patterns
    hidden_layer_outputs = np.maximum(0, input_layer_activations.dot(
        weights_input_hidden) + biases_input_hidden)  # ReLU activation

    # Compute softmax activations for output layer
    output_layer_linear_combination = hidden_layer_outputs.dot(weights_hidden_output) + biases_hidden_output
    output_layer_outputs = np.exp(output_layer_linear_combination) / np.sum(np.exp(output_layer_linear_combination),
                                                                            axis=1, keepdims=True)

    # Step 6: Compute accuracy
    predictions = nn.predict(patterns)
    current_accuracy = accuracy_score(labels, predictions)

    # Step 7: Visualize and save the neural network with activations
    fig = plt.figure(figsize=(12, 6))

    # Subplot for input pattern
    ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1)
    # Select the current input pattern for the iteration (use modulo to loop over patterns)
    current_pattern = patterns[selected_pattern]
    # Reshape and display the input pattern (assuming it is a 2x2 pattern)
    ax1.imshow(current_pattern.reshape(2, 2), cmap='gray', interpolation='none')
    ax1.set_title("Input Pattern")
    ax1.axis('off')

    # Subplot for neural network visualization
    ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1)
    ax2.axis('off')

    left, right, bottom, top = 0.15, 0.85, 0.1, 0.9

    # Set axis limits to match the margins
    ax2.set_xlim(left - 0.1, right + 0.1)
    ax2.set_ylim(bottom - 0.1, top + 0.1)

    # Draw the neural network on the right subplot
    draw_neural_net(ax2, left, right, bottom, top, [patterns.shape[1], 2, len(np.unique(labels))],
                    [nn.coefs_[0], nn.coefs_[1]],
                    [nn.intercepts_[0], nn.intercepts_[1]],
                    [
                        input_layer_activations[selected_pattern],
                        hidden_layer_outputs[selected_pattern],
                        output_layer_outputs[selected_pattern]
                    ], show_text=True)

    plt.suptitle(
        f"Neural Network Visualization with Activations\nIteration: {iteration} - Accuracy: {current_accuracy:.2f}",
        fontsize=16)
    plt.savefig(f'{frames_dir}neural_network_iteration_{iteration:05d}.png')  # Save each iteration as a separate file
    plt.close(fig)

    print(f"Iteration {iteration}, Accuracy: {current_accuracy:.2f}")

print("Training completed.")

compile_frames_to_video()
clear_frames_directory()
