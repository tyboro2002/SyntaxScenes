import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def mandelbrot(c, max_iter):
    """
    Determine the number of iterations before the Mandelbrot sequence for a complex number c diverges.
    """
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z * z + c
        n += 1
    return n


def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    """
    Generate the Mandelbrot set for a given range and resolution.
    """
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width, height))
    for i in range(width):
        for j in range(height):
            n3[i, j] = mandelbrot(r1[i] + 1j * r2[j], max_iter)
    return n3


# Parameters for the Mandelbrot set
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5
width, height = 800, 800
max_iter = 256
frames = 60  # Number of frames for the zoom animation

# Set the center for the zoom
zoom_center_x = -0.75  # X-coordinate of the zoom center
zoom_center_y = 0.0  # Y-coordinate of the zoom center

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# Initialize the plot with the first frame
mandelbrot_image = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)
im = ax.imshow(mandelbrot_image.T, extent=[xmin, xmax, ymin, ymax], cmap='twilight_shifted', interpolation='bilinear')


# Update function for the animation
def update(frame):
    zoom_factor = 1.5 ** (-frame / frames * 10)  # Adjust the zoom factor for smooth zooming
    new_width = (xmax - xmin) * zoom_factor
    new_height = (ymax - ymin) * zoom_factor
    new_xmin = zoom_center_x - new_width / 2
    new_xmax = zoom_center_x + new_width / 2
    new_ymin = zoom_center_y - new_height / 2
    new_ymax = zoom_center_y + new_height / 2

    mandelbrot_image = mandelbrot_set(new_xmin, new_xmax, new_ymin, new_ymax, width, height, max_iter)
    im.set_array(mandelbrot_image.T)
    im.set_extent([new_xmin, new_xmax, new_ymin, new_ymax])
    ax.set_title(f'Mandelbrot Set (Zoom Level: {frame})')
    return [im]


print("generating animation")
# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)

print("saving animation")
# Save the animation as an MP4 file
ani.save('mandelbrot_zoom.mp4', writer='ffmpeg', fps=1)

# plt.show()
