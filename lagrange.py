from matplotlib import pyplot


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


def interpolate_with_lagrange(points: int, path: str, save: bool = False, plot_function: bool = True):
    with open(path, 'r') as f:
        data = f.readlines()

    interpolation_indices = range(0, len(data), len(data) // points + 1)
    interpolation_data = [(float(data[i].split()[0]), float(data[i].split()[1])) for i in interpolation_indices]

    f = interpolation_function(interpolation_data)

    distance = [float(x) for x, _ in (line.split() for line in data)]
    height = [float(y) for _, y in (line.split() for line in data)]
    interpolated_height = [f(x) for x in distance]

    train_distance = [x for x, _ in interpolation_data]
    train_height = [f(x) for x in train_distance]

    pyplot.semilogy(distance, height, 'b.', label='pełne dane')
    if plot_function:
        pyplot.semilogy(distance, interpolated_height, color='green', label='funkcja interpolująca')
    pyplot.semilogy(train_distance, train_height, 'r.', label='dane do interpolacji')

    pyplot.legend()
    pyplot.ylabel('Wysokość')
    pyplot.xlabel('Odległość')
    pyplot.title('Przybliżenie interpolacją Lagrange\'a, ' + str(len(interpolation_data)) + ' punktow')
    pyplot.suptitle(path)
    pyplot.grid()
    if save:
        filename = path.split('/').pop()
        filename.replace(".txt", "")
        pyplot.savefig(f"charts/{filename.replace('.txt', '')}-{points}-points.png")
    pyplot.show()
