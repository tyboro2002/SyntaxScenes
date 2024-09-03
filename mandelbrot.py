import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def mandelbrot(c, max_iter):
    """
    Vectorized Mandelbrot set calculation.
    """
    z = np.zeros(c.shape, dtype=np.complex128)
    div_time = np.zeros(c.shape, dtype=int)
    mask = np.full(c.shape, True, dtype=bool)

    for i in range(max_iter):
        z[mask] = z[mask] * z[mask] + c[mask]
        mask, old_mask = abs(z) <= 2, mask
        div_time[old_mask & ~mask] = i

    div_time[div_time == 0] = max_iter
    return div_time


def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    """
    Generate the Mandelbrot set using vectorized operations.
    """
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    r1, r2 = np.meshgrid(r1, r2)
    c = r1 + 1j * r2
    return mandelbrot(c, max_iter)


# Parameters for the Mandelbrot set
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5
width, height = 800, 800
max_iter = 256
frames = 800  # Number of frames for the zoom animation
FPS = 20

# Set the center for the zoom
zoom_center_x = -0.743643887037158704752191506114774  # X-coordinate of the zoom center
zoom_center_y = 0.131825904205311970493132056385139  # Y-coordinate of the zoom center

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 12))

# Initialize the plot with the first frame
mandelbrot_image = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)
im = ax.imshow(mandelbrot_image.T, extent=[xmin, xmax, ymin, ymax], cmap='twilight_shifted', interpolation='bilinear')


# Update function for the animation
def update(frame):
    print(f"creating frame {frame}")
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
    ax.axis('off')  # Hide the axis
    return [im]


print("generating animation")
# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)

print("saving animation")
# Save the animation as an MP4 file
ani.save('mandelbrot_zoom.mp4', writer='ffmpeg', fps=FPS)
