from matplotlib import pyplot


def draw_map(path: str, save: bool = False):
    with open(path, 'r') as f:
        data = f.readlines()

    distance = [float(x) for x, _ in (line.split() for line in data)]
    height = [float(y) for _, y in (line.split() for line in data)]

    filename = path.split('/').pop()
    filename = filename.replace(".txt", "")
    pyplot.plot(distance, height, '-')
    pyplot.legend()
    pyplot.ylabel('Wysokość [m]')
    pyplot.xlabel('Odległość [m]')
    pyplot.title(f'Mapa dla trasy {filename}')
    pyplot.grid()
    if save:
        pyplot.savefig(f"charts/maps/{filename}-map.png")
    pyplot.show()
