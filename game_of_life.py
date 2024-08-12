import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Get custom rules from the user
# rule = input("Enter the rules (e.g., '23/3' for Conway's Game of Life): ")
rule = "23/3"
survival_str, birth_str = rule.split('/')
survival = [int(n) for n in survival_str]
birth = [int(n) for n in birth_str]

# Grid size
GRID_SIZE = 100  # Increased size for better visualization on full screen


# Function to parse a grid definition from a multi-line string
def parse_grid_from_string(grid_string, size):
    # Initialize a grid of zeros
    grid = np.zeros((size, size), dtype=int)

    # Split the grid string into lines
    lines = grid_string.strip().split('\n')

    # Determine the number of rows and columns in the pattern
    pattern_rows = len(lines)
    pattern_cols = len(lines[0])

    # Ensure the pattern fits within the grid
    if pattern_rows > size or pattern_cols > size:
        raise ValueError("Pattern size exceeds grid size")

    # Place the pattern into the grid
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            if char == 'O':
                grid[r, c] = 1

    return grid

# Example custom grid (Glider pattern in the top-left corner)
def create_custom_grid(size):
    grid = np.zeros((size, size), dtype=int)
    glider_shooter = parse_grid_from_string(""".O.............................................
..O............................................
OOO............................................
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
.............................................OO
............................................OO.
..............................................O""", 50)
    grid[1:51, 1:51] = glider_shooter
    return grid


# Create a grid of the specified size
def create_grid(size):
    return np.random.choice([0, 1], size * size, p=[0.8, 0.2]).reshape(size, size)


# Count live neighbors for a cell at (x, y)
def count_live_neighbors(grid, x, y):
    total = 0
    for i in range(x - 1, x + 2):
        for j in range(y - 1, y + 2):
            if (i == x and j == y) or i < 0 or j < 0 or i >= GRID_SIZE or j >= GRID_SIZE:
                continue
            total += grid[i, j]
    return total


# Update the grid based on the rules
def update_grid(grid, survival, birth):
    new_grid = grid.copy()
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            live_neighbors = count_live_neighbors(grid, i, j)
            if grid[i, j] == 1:
                if live_neighbors not in survival:
                    new_grid[i, j] = 0
            else:
                if live_neighbors in birth:
                    new_grid[i, j] = 1
    return new_grid


# Initialize the grid
# grid = create_grid(GRID_SIZE)
grid = create_custom_grid(GRID_SIZE)

# Set up the plot
fig, ax = plt.subplots()
ax.axis('off')  # Hide the axis
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
plt.imshow(grid, cmap='binary')
mat = ax.matshow(grid)


def update(data):
    global grid
    mat.set_data(grid)
    grid = update_grid(grid, survival, birth)
    return [mat]


# Animate the plot
ani = animation.FuncAnimation(fig, update, interval=200, save_count=50)

# Save the animation
ani.save('game_of_life.mp4', writer='ffmpeg', fps=5)
# plt.show()
