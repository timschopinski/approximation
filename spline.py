import matplotlib.pyplot as plt
import numpy as np


def interpolate_with_spline(points, path, save=False, plot_function=True, equal_separator: bool = True):
    with open(path, 'r') as f:
        data = f.readlines()

    distance = np.array([float(x) for x, _ in (line.split() for line in data)])
    height = np.array([float(y) for _, y in (line.split() for line in data)])

    n = len(distance)

    h = np.diff(distance)

    delta = np.diff(height)

    matrix = np.zeros((n, n))
    matrix[0, 0] = 1
    matrix[n-1, n-1] = 1

    for i in range(1, n-1):
        matrix[i, i-1] = h[i-1]
        matrix[i, i] = 2 * (h[i-1] + h[i])
        matrix[i, i+1] = h[i]

    vector = np.zeros(n)
    vector[1:n-1] = 3 * (delta[1:] / h[1:] - delta[:-1] / h[:-1])

    c = np.linalg.solve(matrix, vector)

    a = height[:-1]
    b = delta / h - h * c[:-1] / 3
    d = (c[1:] - c[:-1]) / (3 * h)

    if equal_separator:
        interp_indices = np.linspace(0, n-1, points, dtype=int)
    else:
        interp_indices = np.sort(np.random.choice(range(1, n-1), points-2, replace=False))
        interp_indices = np.concatenate(([0], interp_indices, [n-1]))

    x_interp = distance[interp_indices]
    y_interp = np.zeros_like(x_interp)

    for i, x in enumerate(x_interp):
        index = np.searchsorted(distance, x)
        if index == 0:
            index = 1
        elif index == n:
            index = n - 1

        dx = x - distance[index-1]
        y_interp[i] = a[index-1] + b[index-1] * dx + c[index-1] * dx**2 + d[index-1] * dx**3

    plt.semilogy(distance, height, 'b.', label='pełne dane')
    if plot_function:
        plt.semilogy(x_interp, y_interp, color='green', label='funkcja interpolująca')

    plt.semilogy(distance[interp_indices], height[interp_indices], 'r.', label='dane do interpolacji')
    filename = path.split('/')[-1].replace('.txt', '')
    mse = np.mean((y_interp - height[interp_indices]) ** 2)
    print(f"Spline {filename}-{points} MSE: {mse}")
    plt.xlabel('Odległość [m]')
    plt.ylabel('Wysokość [m]')
    plt.title(f'Interpolacja funkcjami sklejanymi trzeciego stopnia, {len(interp_indices)} punktów')
    plt.legend()
    plt.grid()

    if save:
        plt.savefig(f'charts/spline/{filename}-{points}-points.png')

    plt.show()
