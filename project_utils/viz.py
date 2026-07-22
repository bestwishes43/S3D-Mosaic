import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def implay(images, vmax=None, vmin=None, interval=200):
    images = np.transpose(images, [2, 0, 1])

    fig, ax = plt.subplots()
    if vmax is None:
        vmax = np.max(images)
    if vmin is None:
        vmin = np.min(images)
    image_ax = ax.imshow(np.zeros_like(images[0]), animated=True, cmap='gray',vmax=vmax, vmin=vmin)
    ax.set_axis_off()
    def update(idx):
        image_ax.set_data(images[idx])
        return plt.gca(),
    # 创建动画
    ani = FuncAnimation(fig, update, frames=range(len(images)), interval=interval, blit=True)
    plt.show()

def imshow(images):
    for image in images:
        _, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.set_axis_off()
    plt.show()