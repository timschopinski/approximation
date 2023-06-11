import math

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


def interpolate_with_lagrange(k: int, path: str, save: bool = False, log_scale: bool = True):
    with open(path, 'r') as f:
        data = f.readlines()

    # create data for function
    interpolation_data = [(float(x), float(y)) for x, y in (line.split() for line in data[1::k])]

    # use data to create interpolating function F
    f = interpolation_function(interpolation_data)

    distance = [float(x) for x, _ in (line.split() for line in data)]
    height = [float(y) for _, y in (line.split() for line in data)]
    interpolated_height = [f(x) for x in distance]

    train_distance = [x for x, _ in interpolation_data]
    train_height = [f(x) for x in train_distance]

    # odkomentowanie poniższych funkcji pozwoli na wyświetlenie fragmentów aproksymacji bez oscylacji
    #
    n = math.floor(len(distance)/3)

    if log_scale:
        pyplot.semilogy(distance, height, 'r.', label='pełne dane')
        pyplot.semilogy(distance, interpolated_height, color='blue', label='funkcja interpolująca')
        pyplot.semilogy(train_distance, train_height, 'g.', label='dane do interpolacji')
    else:
        pyplot.plot(distance[n:2*n], height[n:2*n], 'r.', label='pełne dane')
        pyplot.plot(distance[n:2*n], interpolated_height[n:2*n], color='blue', label='funkcja interpolująca')
        pyplot.plot(train_distance[n:2*n], train_height[n:2*n], 'g.', label='dane do interpolacji')
    pyplot.legend()
    pyplot.ylabel('Wysokość')
    pyplot.xlabel('Odległość')
    pyplot.title('Przybliżenie interpolacją Lagrange\'a, ' + str(len(interpolation_data)) + ' punkty(ów)')
    pyplot.suptitle(path)
    pyplot.grid()
    if save:
        filename = path.split('/').pop()
        filename.replace(".txt", "")
        filename += "-log" if log_scale else ""
        pyplot.savefig(f"charts/{filename.replace('.txt', '')}-{k}-points.png")
    pyplot.show()
