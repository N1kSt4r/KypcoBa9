import numpy as np
import matplotlib.pyplot as plt
import pywt

haar_wavelet = lambda x: 0 if x < 0 else 1 if x < 0.5 else -1 if x <= 1 else 0

fi = lambda m, k, t: haar_wavelet(t* 2 ** m - k) * 2 ** (m / 2)
frequency = 1000
N = 1024

def get_values(m, k, x):
    values = np.array(list(map(lambda t: fi(m, k, t), x)))
    values[np.abs(values) < 1e-5] = None
    return values

x = np.linspace(0, 1, 512)
f, sub = plt.subplots(1, 3, sharey=True)

plt.figure(dpi=125)
sub[0].plot(x, get_values(0, 0, x))
sub[1].plot(x, get_values(1, 0, x))
sub[1].plot(x, get_values(1, 1, x))
sub[2].plot(x, get_values(2, 0, x))
sub[2].plot(x, get_values(2, 1, x))
sub[2].plot(x, get_values(2, 2, x))
sub[2].plot(x, get_values(2, 3, x))
f.show()
# plt.plot(x, list(map(lambda t: fi(1, 1, t), x)), 'b')
# plt.plot(x, list(map(lambda t: fi(1, 2, t), x)), 'b')
# plt.plot(x, list(map(lambda t: fi(2, 0, t), x)), 'r')
# plt.plot(x, list(map(lambda t: fi(2, 1, t), x)), 'r')
# plt.plot(x, list(map(lambda t: fi(3, 0, t), x)), 'g')
# plt.plot(x, list(map(lambda t: fi(3, 1, t), x)), 'g')
# plt.plot(x, list(map(lambda t: fi(3, 2, t), x)), 'g')
# plt.plot(x, list(map(lambda t: fi(3, 3, t), x)), 'g')


def sin_freq(t, freq):
    return np.sin(2 * np.pi * freq * t / frequency)
signal = np.array([sin_freq(t, 44) + 2 * sin_freq(t, 20)
                   for t in range(N)])

level = 8
indexes = np.arange(len(signal))
d = np.zeros((level, 2**level))
for j in range(level):
    for k in range(2**j):
        for i in range(len(signal)):
            d[j, k] += fi(j, k, i) * signal[i]

x2 = 0

for j in range(level):
    for k in range(2**j):
        x2 += d[j, k] * fi(j, k, 23)
print(x2, signal[23])