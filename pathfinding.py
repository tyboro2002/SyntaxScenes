import matplotlib.pyplot as plt
import matplotlib.animation as animation
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from collections import deque
from moviepy.editor import VideoFileClip, concatenate_videoclips

display_country_names = True
exploration_animation_filename = "exploration_animation.mp4"
path_animation_filename = "final_path_animation.mp4"
concatenated_animation_filename = "combined_animation.mp4"

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

# Define the countries for the animation
start_country = 'Ukraine'
end_country = 'Portugal'
path, explored_nodes = bfs(start_country, end_country, adjacency_list)

# Set up the plot for the exploration animation
fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_title(f'Exploration Animation: Path Between {start_country} and {end_country}')
ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
ax.add_feature(cfeature.COASTLINE)
world.boundary.plot(ax=ax, linewidth=1)
ax.set_extent([-30, 60, 35, 75], crs=ccrs.PlateCarree())
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

for country, (lon, lat) in locations.items():
    ax.plot(lon, lat, marker='o', color='red', markersize=6)
    if display_country_names:
        ax.text(lon, lat, country, fontsize=8, ha='center', color='black')

lines = [ax.plot([], [], color='blue', linewidth=2, linestyle='--')[0] for _ in range(len(explored_nodes) - 1)]

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def update_exploration(num):
    if num < len(explored_nodes):
        for i, line in enumerate(lines):
            if i < num:
                start = explored_nodes[i]
                end = explored_nodes[i + 1]
                x_values = [locations[start][0], locations[end][0]]
                y_values = [locations[start][1], locations[end][1]]
                line.set_data(x_values, y_values)
    return lines

exploration_ani = animation.FuncAnimation(fig, update_exploration, frames=len(explored_nodes)+1, init_func=init, blit=True, repeat=False, interval=1000)
exploration_ani.save(exploration_animation_filename, writer='ffmpeg', fps=1)
print(f"Exploration animation saved successfully as {exploration_animation_filename}")

# Set up the plot for the final path animation
fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_title(f'Final Path Animation: Path Between {start_country} and {end_country}')
ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
ax.add_feature(cfeature.COASTLINE)
world.boundary.plot(ax=ax, linewidth=1)
ax.set_extent([-30, 60, 35, 75], crs=ccrs.PlateCarree())
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

for country, (lon, lat) in locations.items():
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
final_ani = animation.FuncAnimation(fig, update_final, frames=len(path)+1, init_func=init_final, blit=True, repeat=False, interval=1000)
final_ani.save(path_animation_filename, writer='ffmpeg', fps=1)
print(f"Final path animation saved successfully as {path_animation_filename}")

# Concatenate the two animations
clip1 = VideoFileClip(exploration_animation_filename)
clip2 = VideoFileClip(path_animation_filename)

# Combine the two clips
final_clip = concatenate_videoclips([clip1, clip2])
final_clip.write_videofile(concatenated_animation_filename, fps=24)

print(f"Combined animation saved successfully as {concatenated_animation_filename}")
