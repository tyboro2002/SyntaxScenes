import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

# Define a custom colormap
colors = ["black", "Yellow"]
n_bins = 100  # Number of bins for the gradient
cmap_name = "yellow_black"

# Create the colormap
cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Get custom rules from the user
# rule = input("Enter the rules (e.g., '23/3' for Conway's Game of Life): ")
rule = "23/3"
amount_of_frames = 150
pattern = "glider_shooter"  # glider_shooter, toad, beacon, glider, pulsar, pentadecathlon or random
FPS = 8  # normally 5
survival_str, birth_str = rule.split('/')
survival = [int(n) for n in survival_str]
birth = [int(n) for n in birth_str]

# Grid dimensions
GRID_WIDTH = 50  # Width of the grid (50)
GRID_HEIGHT = 90  # Height of the grid (90)

X_OFFSET = 30  # 30
Y_OFFSET = 1  # 1


# Function to parse a grid definition from a multi-line string
def parse_grid_from_string(grid_string, width, height):
    """
    Parse a grid definition from a multi-line string and return a numpy array representation.

    Args:
        grid_string (str): A multi-line string defining the grid pattern with 'O' for live cells and '.' for empty space.
        width (int): The width of the grid to fit the pattern into.
        height (int): The height of the grid to fit the pattern into.

    Returns:
        numpy.ndarray: A 2D numpy array representing the parsed grid.
    """
    grid = np.zeros((height, width), dtype=int)  # Initialize a grid of zeros

    # Split the grid string into lines
    lines = grid_string.strip().split('\n')

    # Determine the number of rows and columns in the pattern
    pattern_rows = len(lines)
    pattern_cols = len(lines[0])
    print("shape is of size (rows, cols): ")
    print(pattern_rows, pattern_cols)

    # Ensure the pattern fits within the grid
    if pattern_rows > height or pattern_cols > width:
        raise ValueError("Pattern size exceeds grid size")

    # Place the pattern into the grid
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            if char == 'O':
                grid[r, c] = 1  # Mark the cell as alive

    return grid


# Create a custom grid based on a specified pattern or generate a random grid.
def create_custom_grid(width, height, x_offset=1, y_offset=1, pattern="random"):
    """
    Create a custom grid based on a specified pattern or generate a random grid.

    Args:
        width (int): The width of the grid to create.
        height (int): The height of the grid to create.
        pattern (str): The name of the pattern to use or "random" for a random grid.
        x_offset (int): The x offset
        y_offset (int): The y offset

    Returns:
        numpy.ndarray: A 2D numpy array representing the grid.
    """
    grid = np.zeros((height, width), dtype=int)  # Initialize an empty grid

    # Define patterns
    if pattern == "glider_shooter":
        glider_shooter = parse_grid_from_string("""
...............................................
...............................................
...............................................
...............................................
...............................................
...............................................
...............................................
...............................................
...............O...............................
...............OOOO............................
................OOOO..........OO...............
.....OO.........O..O.........O.O...............
.....OO.........OOOO........OOO........OO......
...............OOOO........OOO.........OO......
...............O............OOO................
.............................O.O...............
..............................OO...............
...............................................
...............................................
...............................................
...............................................
...............................................
...............................................
...............................................
...............................................
...............................................""", 47, 26)
        grid[x_offset+0:x_offset+26, y_offset+0:y_offset+47] = glider_shooter  # Place the pattern in the grid

    elif pattern == "toad":
        toad = parse_grid_from_string("""
............
.....OOO.....
....OOO......
............
""", 10, 4)
        grid[x_offset+0:x_offset+4, y_offset+0:y_offset+10] = toad  # Place the pattern in the grid

    elif pattern == "beacon":
        beacon = parse_grid_from_string("""
OO..
OO..
..OO
..OO
""", 4, 4)
        grid[x_offset+0:x_offset+4, y_offset+0:y_offset+4] = beacon  # Place the pattern in the grid

    elif pattern == "glider":
        glider = parse_grid_from_string("""
.O.
..O
OOO
""", 3, 3)
        grid[x_offset+0:x_offset+3, y_offset+0:y_offset+3] = glider  # Place the pattern in the grid

    elif pattern == "pulsar":
        pulsar = parse_grid_from_string("""
..OOO...OOO..
.............
O....O.O....O
O....O.O....O
O....O.O....O
..OOO...OOO..
.............
..OOO...OOO..
O....O.O....O
O....O.O....O
O....O.O....O
.............
..OOO...OOO..
""", 13, 13)
        grid[x_offset+0:x_offset+13, y_offset+0:y_offset+13] = pulsar  # Place the pattern in the grid

    elif pattern == "pentadecathlon":
        pentadecathlon = parse_grid_from_string("""
.............
....O....O....
....O....O....
....O....O....
....O....O....
....O....O....
....O....O....
.............
""", 13, 8)
        grid[x_offset+0:x_offset+8, y_offset+0:y_offset+13] = pentadecathlon  # Place the pattern in the grid

    elif pattern == "random":
        # Generate a random grid with live and dead cells
        grid = np.random.choice([0, 1], width * height, p=[0.8, 0.2]).reshape(height, width)

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return grid


# Count live neighbors for a cell at (x, y)
def count_live_neighbors(grid, x, y):
    total = 0
    for i in range(x - 1, x + 2):
        for j in range(y - 1, y + 2):
            if (i == x and j == y) or i < 0 or j < 0 or i >= GRID_HEIGHT or j >= GRID_WIDTH:
                continue
            total += grid[i, j]
    return total


# Update the grid based on the rules
def update_grid(grid, survival, birth):
    new_grid = grid.copy()
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            live_neighbors = count_live_neighbors(grid, i, j)
            if grid[i, j] == 1:
                if live_neighbors not in survival:
                    new_grid[i, j] = 0
            else:
                if live_neighbors in birth:
                    new_grid[i, j] = 1
    return new_grid


# Initialize the grid
grid = create_custom_grid(GRID_WIDTH, GRID_HEIGHT, x_offset=X_OFFSET, y_offset=Y_OFFSET, pattern=pattern)

# Set up the plot
fig, ax = plt.subplots(figsize=(9, 16))  # Adjusted figure size for vertical format
ax.axis('off')  # Hide the axis
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
mat = ax.matshow(grid, cmap=cmap)


def update(data):
    global grid
    if data == 0:
        # Show the initial grid
        mat.set_data(grid)
    else:
        # Update the grid for the animation
        grid = update_grid(grid, survival, birth)
        mat.set_data(grid)
    return [mat]


# Animate the plot
ani = animation.FuncAnimation(fig, update, frames=amount_of_frames, interval=1000 // FPS, repeat=False)

# Save the animation
ani.save('game_of_life.mp4', writer='ffmpeg', fps=FPS)
