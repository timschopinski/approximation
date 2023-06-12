from lagrange import interpolate_with_lagrange
from spline import interpolate_with_spline
from map import draw_map


def main():
    paths = [
        "2018_paths/tczew_starogard.txt",
        "2018_paths/ulm_lugano.txt",
        "2018_paths/genoa_rapallo.txt"
    ]
    points = [3, 6, 18]
    for n_points in points:
        for path in paths:
            draw_map(path, save=True)
            interpolate_with_lagrange(n_points, path, save=True, plot_function=True, equal_separator=True)
            interpolate_with_spline(n_points, path, save=True, plot_function=True, equal_separator=False)


if __name__ == '__main__':
    main()
