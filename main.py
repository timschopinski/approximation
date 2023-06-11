from lagrange import interpolate_with_lagrange
from spline import interpolate_with_spline

interpolate_with_lagrange(6, '2018_paths/tczew_starogard.txt', save=False, plot_function=True)
interpolate_with_spline(6, '2018_paths/tczew_starogard.txt', save=False, plot_function=True)
