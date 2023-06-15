import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def interpolate_with_spline(points, path, save=False, plot_function=True, equal_separator=True):
    with open(path, 'r') as f:
        data = f.readlines()

    distance = np.array([float(x) for x, _ in (line.split() for line in data)])
    height = np.array([float(y) for _, y in (line.split() for line in data)])

    if equal_separator:
        interp_indices = np.linspace(0, len(distance) - 1, points, dtype=int)
    else:
        interp_indices = np.sort(np.random.choice(range(len(distance)), points-2, replace=False))
        interp_indices = np.concatenate(([0], interp_indices, [len(distance)-1]))

    x_interp = distance[interp_indices]
    y_interp = height[interp_indices]

    f = interp1d(distance, height, kind='cubic')
    y_smooth = f(x_interp)

    plt.semilogy(distance, height, 'b.', label='pełne dane')
    if plot_function:
        plt.semilogy(x_interp, y_smooth, color='green', label='funkcja interpolująca')
    plt.semilogy(distance[interp_indices], y_interp, 'r.', label='dane do interpolacji')
    filename = path.split('/')[-1].replace('.txt', '')
    mse = np.mean((y_smooth - y_interp) ** 2)
    print(f"Spline {filename}-{points} MSE: {mse}")
    plt.xlabel('Odległość [m]')
    plt.ylabel('Wysokość [m]')
    plt.title(f'Interpolacja funkcjami sklejanymi trzeciego stopnia, {len(interp_indices)} punktów')
    plt.legend()
    plt.grid()

    if save:
        plt.savefig(f'charts/spline/{filename}-{points}-points.png')

    plt.show()
