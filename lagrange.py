import random
from matplotlib import pyplot
import numpy as np


def interpolation_function(points):
    n = len(points)

    def evaluate(x):
        result = 0
        first_x, last_x = points[0][0], points[-1][0]
        if x < first_x or x > last_x:
            return 0
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

    return evaluate


def interpolate_with_lagrange(points: int, path: str, save: bool = False, plot_function: bool = True, equal_separator: bool = True):
    with open(path, 'r') as f:
        data = f.readlines()
    n = len(data)
    if equal_separator:
        interpolation_indices = [int(i * (n - 1) / (points - 1)) for i in range(points)]
    else:
        interpolation_indices = [random.choice(range(1, n - 1)) for _ in range(points - 2)]
        interpolation_indices = [0] + sorted(interpolation_indices) + [n - 1]
    interpolation_data = [(float(data[i].split()[0]), float(data[i].split()[1])) for i in interpolation_indices]

    f = interpolation_function(interpolation_data)

    distance = np.array([float(x) for x, _ in (line.split() for line in data)])
    height = np.array([float(y) for _, y in (line.split() for line in data)])
    interpolated_height = np.array([f(x) for x in distance])

    train_distance = np.array([x for x, _ in interpolation_data])
    train_height = np.array([f(x) for x in train_distance])

    true_height = np.array([float(y) for _, y in (line.split() for line in data)])

    # Obliczanie wysokości dla całego zakresu odległości
    interpolated_distance = np.linspace(distance[0], distance[-1], num=1000)
    interpolated_height = np.array([f(x) for x in interpolated_distance])

    # Usuwanie punktów o zerowej wysokości z interpolowanych danych
    mask = interpolated_height != 0
    interpolated_distance = interpolated_distance[mask]
    interpolated_height = interpolated_height[mask]

    filename = path.split('/')[-1].replace('.txt', '')
    pyplot.semilogy(distance, height, 'b.', label='pełne dane')
    if plot_function:
        pyplot.semilogy(interpolated_distance, interpolated_height, color='green', label='funkcja interpolująca')
    pyplot.semilogy(train_distance, train_height, 'r.', label='dane do interpolacji')

    pyplot.legend()
    pyplot.ylabel('Wysokość [m]')
    pyplot.xlabel('Odległość [m]')
    pyplot.title(f'Interpolacja Lagrange\'a dla {len(interpolation_data)} punktów\n')
    pyplot.grid()
    if save:
        pyplot.savefig(f"charts/lagrange/{filename}-{points}-points.png")
    pyplot.show()
