import matplotlib.pyplot as plt
import matplotlib.animation as animation
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from collections import deque
from moviepy.editor import VideoFileClip, concatenate_videoclips
import heapq
import math
from collections import deque

# Define the countries for the animation
start_country = 'Ukraine'
end_country = 'Portugal'
algorithm = "astar"  # supports "dfs", "bfs", "dijkstra", "astar"
display_country_names = False
exploration_animation_filename = "pathfinding/exploration_animation.mp4"
path_animation_filename = "pathfinding/final_path_animation.mp4"
concatenated_animation_filename = \
    f'pathfinding/Final_Path_Animation_Path_Between_{start_country}_and_{end_country}_using_{algorithm}.mp4'
exploration_fps = 2
path_fps = 1

# Path to the shapefile
shapefile_path = "assets/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"

# Load the shapefile
world = gpd.read_file(shapefile_path)

# Reproject to WGS84 (EPSG:4326) for latitude/longitude
world = world.to_crs(epsg=4326)

# Create a dictionary of country names and their coordinates
locations = {}
for idx, row in world.iterrows():
    country = row['SOVEREIGNT']
    centroid = row['geometry'].centroid
    locations[country] = (centroid.x, centroid.y)

# Correct specific coordinates
locations['France'] = (2.2137, 46.2276)
locations['United States'] = (-101.299591, 40.116386)

# Create an adjacency list for countries
adjacency_list = {country: set() for country in world['SOVEREIGNT']}
for idx, row in world.iterrows():
    country = row['SOVEREIGNT']
    geometry = row['geometry']
    for other_idx, other_row in world.iterrows():
        other_country = other_row['SOVEREIGNT']
        if country != other_country:
            other_geometry = other_row['geometry']
            if geometry.intersects(other_geometry):
                adjacency_list[country].add(other_country)


def heuristic(a, b):
    """Calculate the straight-line (Euclidean) distance between two points."""
    lon1, lat1 = locations[a]
    lon2, lat2 = locations[b]
    return math.sqrt((lon2 - lon1) ** 2 + (lat2 - lat1) ** 2)


def astar(start, goal, adjacency_list):
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))  # (f, g, current_node, path)
    came_from = {}
    g_score = {start: 0}
    explored_nodes = []

    while open_set:
        _, current_g, current, path = heapq.heappop(open_set)
        explored_nodes.append(current)

        if current == goal:
            return path, explored_nodes

        for neighbor in adjacency_list[current]:
            tentative_g_score = current_g + heuristic(current, neighbor)  # g_score from start to neighbor

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g_score, neighbor, path + [neighbor]))
                came_from[neighbor] = current

    return None, explored_nodes


# BFS function to find the shortest path and keep track of explored nodes
def bfs(start, goal, adjacency_list):
    queue = deque([(start, [start])])
    visited = set()
    explored_nodes = []

    while queue:
        vertex, path = queue.popleft()
        if vertex in visited:
            continue
        visited.add(vertex)
        explored_nodes.append(vertex)

        for neighbor in adjacency_list[vertex]:
            if neighbor in visited:
                continue
            new_path = path + [neighbor]
            if neighbor == goal:
                explored_nodes.append(neighbor)
                return new_path, explored_nodes
            queue.append((neighbor, new_path))

    return None, explored_nodes


def bfs(start, goal, adjacency_list):
    queue = deque([(start, [start])])
    visited = set()
    explored_nodes = []

    while queue:
        vertex, path = queue.popleft()
        if vertex in visited:
            continue
        visited.add(vertex)
        explored_nodes.append(vertex)

        if vertex == goal:
            return path, explored_nodes

        for neighbor in adjacency_list[vertex]:
            if neighbor not in visited:
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))

    return None, explored_nodes


def dijkstra(start, goal, adjacency_list):
    # Priority queue to store (cost, current_node, path)
    open_set = []
    heapq.heappush(open_set, (0, start, [start]))  # (cost, current_node, path)

    # Distance from start to each node
    distances = {node: float('inf') for node in adjacency_list}
    distances[start] = 0

    # Path tracking
    came_from = {}
    explored_nodes = []

    while open_set:
        current_cost, current_node, path = heapq.heappop(open_set)
        explored_nodes.append(current_node)

        if current_node == goal:
            return path, explored_nodes

        for neighbor in adjacency_list[current_node]:
            # Assume each edge has the same weight for simplicity
            edge_cost = heuristic(current_node, neighbor)  # Using heuristic as a stand-in for edge weight
            new_cost = current_cost + edge_cost

            if new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                heapq.heappush(open_set, (new_cost, neighbor, path + [neighbor]))
                came_from[neighbor] = current_node

    return None, explored_nodes


def dfs(start, goal, adjacency_list):
    stack = [(start, [start])]  # Stack to store (current_node, path)
    visited = set()
    explored_nodes = []

    while stack:
        vertex, path = stack.pop()
        if vertex in visited:
            continue
        visited.add(vertex)
        explored_nodes.append(vertex)

        if vertex == goal:
            return path, explored_nodes

        for neighbor in adjacency_list[vertex]:
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))

    return None, explored_nodes


def calculate_path_length(path):
    """Calculate the total length of the path."""
    if not path:
        return 0
    total_length = 0
    for i in range(len(path) - 1):
        lon1, lat1 = locations[path[i]]
        lon2, lat2 = locations[path[i + 1]]
        total_length += math.sqrt((lon2 - lon1) ** 2 + (lat2 - lat1) ** 2)
    return total_length


if algorithm == "bfs":
    path, explored_nodes = bfs(start_country, end_country, adjacency_list)
if algorithm == "astar":
    path, explored_nodes = astar(start_country, end_country, adjacency_list)
if algorithm == "dijkstra":
    path, explored_nodes = dijkstra(start_country, end_country, adjacency_list)
if algorithm == "dfs":
    path, explored_nodes = dfs(start_country, end_country, adjacency_list)

# Calculate path length
path_length = calculate_path_length(path)

# Set up the plot for the exploration animation
fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
# ax.set_title(f'Exploration Animation: Path Between {start_country} and {end_country}')
ax.axis('off')  # Hide the axis
ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
ax.add_feature(cfeature.COASTLINE)
world.boundary.plot(ax=ax, linewidth=1)

ax.set_extent([-30, 60, 35, 75], crs=ccrs.PlateCarree())
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


# -30, 60, 35, 75 # europe
# -180, 180, -90, 90 # full world


# Plot countries and their names only if they are within the map extent
def is_country_within_extent(country, extent):
    lon, lat = locations[country]
    return extent[0] <= lon <= extent[1] and extent[2] <= lat <= extent[3]


# Get map extent
extent = ax.get_extent(crs=ccrs.PlateCarree())

for country, (lon, lat) in locations.items():
    if is_country_within_extent(country, extent):
        ax.plot(lon, lat, marker='o', color='red', markersize=6)
        if display_country_names:
            ax.text(lon, lat, country, fontsize=8, ha='center', color='black')

# Create line plot objects for the web connections
lines = []
for i in range(len(explored_nodes)):
    for j in range(i + 1, len(explored_nodes)):
        if explored_nodes[j] in adjacency_list[explored_nodes[i]]:
            line, = ax.plot([], [], color='blue', linewidth=2, linestyle='--', alpha=0.5)
            lines.append((line, explored_nodes[i], explored_nodes[j]))


def init():
    for line, _, _ in lines:
        line.set_data([], [])
    return [line for line, _, _ in lines]


def update_exploration(num):
    for line, start, end in lines:
        start_idx = explored_nodes.index(start)
        end_idx = explored_nodes.index(end)
        if start_idx <= num and end_idx <= num:
            x_values = [locations[start][0], locations[end][0]]
            y_values = [locations[start][1], locations[end][1]]
            line.set_data(x_values, y_values)
    return [line for line, _, _ in lines]


exploration_ani = animation.FuncAnimation(fig, update_exploration, frames=len(explored_nodes), init_func=init,
                                          blit=True, repeat=False, interval=1000)
exploration_ani.save(exploration_animation_filename, writer='ffmpeg', fps=exploration_fps)
print(f"Exploration animation saved successfully as {exploration_animation_filename}")

# Set up the plot for the final path animation
fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
# ax.set_title(f'Final Path Animation: Path Between {start_country} and {end_country}')
ax.set_title(f'Path of length {path_length:.2f}')
ax.axis('off')  # Hide the axis
ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
ax.add_feature(cfeature.COASTLINE)
world.boundary.plot(ax=ax, linewidth=1)
ax.set_extent([-30, 60, 35, 75], crs=ccrs.PlateCarree())
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


def is_country_within_extent(country, extent):
    lon, lat = locations[country]
    return extent[0] <= lon <= extent[1] and extent[2] <= lat <= extent[3]


# Get map extent
extent = ax.get_extent(crs=ccrs.PlateCarree())

for country, (lon, lat) in locations.items():
    if is_country_within_extent(country, extent):
        ax.plot(lon, lat, marker='o', color='red', markersize=6)
        if display_country_names:
            ax.text(lon, lat, country, fontsize=8, ha='center', color='black')

final_lines = [ax.plot([], [], color='green', linewidth=2, linestyle='-')[0] for _ in range(len(path) - 1)]


def init_final():
    for line in final_lines:
        line.set_data([], [])
    return final_lines


def update_final(num):
    if num < len(path):
        for i, line in enumerate(final_lines):
            if i < num:
                start = path[i]
                end = path[i + 1]
                x_values = [locations[start][0], locations[end][0]]
                y_values = [locations[start][1], locations[end][1]]
                line.set_data(x_values, y_values)
    return final_lines


print(path)
final_ani = animation.FuncAnimation(fig, update_final, frames=len(path), init_func=init_final, blit=True, repeat=False,
                                    interval=1000)
final_ani.save(path_animation_filename, writer='ffmpeg', fps=path_fps)
print(f"Final path animation saved successfully as {path_animation_filename}")

# Concatenate the two animations
clip1 = VideoFileClip(exploration_animation_filename)
clip2 = VideoFileClip(path_animation_filename)

# Combine the two clips
final_clip = concatenate_videoclips([clip1, clip2])
final_clip.write_videofile(concatenated_animation_filename, fps=24)

print(f"Combined animation saved successfully as {concatenated_animation_filename}")
