import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


class wavelet_series:
    def __init__(self, g, mother_wavelet, levels = 5):
        self.levels = levels
        self.scaling = lambda x: 0 if x < 0 else 0 if x >= 1 else 1

        self.basis = [[(lambda i, j: lambda x: 2 ** (i / 2) * mother_wavelet(2 ** i * x - j))(i, j)
                       for j in range(2 ** i)] for i in range(levels)]

        self.coef = [[integrate.quad(lambda x: self.basis[i][j](x) * g(x), 0, 1)[0]
                      for j in range(2 ** i)] for i in range(levels)]

        self.scaling_coef = [integrate.quad(lambda x: g(x) * self.scaling(x - i), -np.inf, np.inf)[0]
                             for i in range(levels)]

    def __call__(self, point):
        value = 0
        for i in range(self.levels):
            for j in range(2 ** i):
                value += self.coef[i][j] * self.basis[i][j](point)
            #value += self.scaling_coef[i] * self.scaling(point - i)
        return value

a = 0
b = 1
g = lambda x: x**2 + 3


xs = np.linspace(a+1e-1, b-1e-1, 1000)

haar_wavelet = lambda x: 0 if x < 0 else 1 if x < 0.5 else -1 if x <= 1 else 0
f = wavelet_series(g, haar_wavelet)
plt.plot(xs, list(map(f, xs)))

morle_wavelet = lambda x: np.exp(-x**2/2)*np.cos(5*x)
f = wavelet_series(g, morle_wavelet)
plt.plot(xs, list(map(f, xs)))

import pywt
# plt.plot(xs[::2], pywt.dwt(list(map(g, xs)), 'haar')[0])

plt.plot(xs, list(map(g, xs)))
plt.show()