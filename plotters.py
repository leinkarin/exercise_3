import matplotlib.pyplot as plt
import numpy as np


def plot_graph(plots_values, x_axis, title, x_label, y_label):
    for plot, label in plots_values:
        plt.plot(x_axis, plot, marker='o', label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_graph_7(vectors):
    # extract the x and y coordinates from the vectors
    x_coords = vectors[:, 0]
    y_coords = vectors[:, 1]

    magnitudes = np.linalg.norm(vectors, axis=1)
    normalized_magnitudes = (magnitudes - np.min(magnitudes)) / (np.max(magnitudes) - np.min(magnitudes))
    alphas = 0.2 + 0.5 * normalized_magnitudes

    # plot the points
    plt.scatter(x_coords, y_coords, color='blue', label='Vectors', alpha=alphas)

    for i in range(len(vectors) - 1):
        plt.plot([x_coords[i], x_coords[i + 1]], [y_coords[i], y_coords[i + 1]], color='blue', linestyle='--')

    # add labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimized vector through time')

    plt.legend()
    plt.show()

