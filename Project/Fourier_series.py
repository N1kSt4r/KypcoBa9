import sympy as sp
import numpy as np
from sympy.utilities import lambdify
from scipy import integrate
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


class Fourier_series:
    def __init__(self, function, lower_bound, upper_bound, segments=1000):
        self.segments = segments
        self.a_coef = np.empty(segments, dtype=float)
        self.b_coef = np.empty(segments, dtype=float)
        self.shift_coef = (upper_bound - lower_bound) / 2 / np.pi

        x, n = sp.symbols('x n')
        a = lambdify((x, n), sp.integrate(function(x) * sp.cos(n * x / self.shift_coef), x))
        b = lambdify((x, n), sp.integrate(function(x) * sp.sin(n * x / self.shift_coef), x))

        def get_coef(I, n, a, b):
            return I(b, n) - I(a, n)

        self.a_coef[0] = get_coef(a, 0, lower_bound, upper_bound)
        for k in range(1, segments):
            self.a_coef[k] = get_coef(a, k, lower_bound, upper_bound)
            self.b_coef[k] = get_coef(b, k, lower_bound, upper_bound)

    def __call__(self, point):
        series_sum = self.a_coef[0] / 2
        for k in range(1, self.segments):
            series_sum += self.a_coef[k] * np.cos(k * point / self.shift_coef)
            series_sum += self.b_coef[k] * np.sin(k * point / self.shift_coef)
        return series_sum / self.shift_coef / np.pi


g = lambda x: x**3
f = Fourier_series(g, -1, 1, 100)
xs = np.linspace(-2, 2, 1000)

plt.plot(xs, list(map(f, xs)), 'b')
plt.plot(xs, list(map(g, xs)), 'r')
plt.show()

xs = np.linspace(-0.5, 0.5, 1000)

plt.plot(xs, list(map(f, xs)), 'b')
plt.plot(xs, list(map(g, xs)), 'r')
plt.show()
