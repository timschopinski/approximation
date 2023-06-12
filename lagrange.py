import random
from matplotlib import pyplot
import numpy as np


def interpolation_function(points):
    def f(x):
        result = 0
        n = len(points)
        for i in range(n):
            xi, yi = points[i]
            base = 1
            for j in range(n):
                if i == j:
                    continue
                else:
                    xj, yj = points[j]
                    base *= (float(x) - float(xj)) / float(float(xi) - float(xj))
            result += float(yi) * base
        return result

    return f


def interpolate_with_lagrange(points: int, path: str, save: bool = False, plot_function: bool = True, equal_separator: bool = True):
    with open(path, 'r') as f:
        data = f.readlines()
    n = len(data)
    if equal_separator:
        interpolation_indices = range(0, len(data), len(data) // points + 1)
    else:
        interpolation_indices = [random.choice(range(1, n)) for _ in range(points)]
    interpolation_data = [(float(data[i].split()[0]), float(data[i].split()[1])) for i in interpolation_indices]

    f = interpolation_function(interpolation_data)

    distance = np.array([float(x) for x, _ in (line.split() for line in data)])
    height = np.array([float(y) for _, y in (line.split() for line in data)])
    interpolated_height = np.array([f(x) for x in distance])

    train_distance = np.array([x for x, _ in interpolation_data])
    train_height = np.array([f(x) for x in train_distance])

    true_height = np.array([float(y) for _, y in (line.split() for line in data)])

    mse = np.mean((height - interpolated_height) ** 2)
    filename = path.split('/')[-1].replace('.txt', '')
    print(f"Lagrande {filename}-{points} MSE: {mse}")
    pyplot.semilogy(distance, height, 'b.', label='pełne dane')
    if plot_function:
        pyplot.semilogy(distance, interpolated_height, color='green', label='funkcja interpolująca')
    pyplot.semilogy(train_distance, train_height, 'r.', label='dane do interpolacji')

    pyplot.legend()
    pyplot.ylabel('Wysokość [m]')
    pyplot.xlabel('Odległość [m]')
    pyplot.title(f'Interpolacja Lagrange\'a dla {len(interpolation_data)} punktów\nMSE: {mse:.4f}')
    pyplot.grid()
    if save:
        pyplot.savefig(f"charts/lagrange/{filename}-{points}-points.png")
    pyplot.show()
