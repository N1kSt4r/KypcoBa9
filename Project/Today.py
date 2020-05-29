import numpy as np
import matplotlib.pyplot as plt
import pywt

wavelet = pywt.Wavelet('haar')

# print(wavelet.wavefun(level=2))
# plt.plot(wavelet.wavefun(level=5)[0], 'b')  # scaling
# plt.plot(wavelet.wavefun(level=5)[1], 'r')  # wavelet
# plt.show()

frequency = 1000
# в сигнале частоты от 0 до 100
N = 1024
# количество данных, 1 секунды для такой частоты

haar_wavelet = lambda x: 0 if x < 0 else 1 if x < 0.5 else -1 if x <= 1 else 0

fi = lambda j, k, t: haar_wavelet(t / 2**(j + 1) - k) / (j + 1)**0.5
# fi = lambda j, k, t: 2**(j / 2) * haar_wavelet(t * 2**j - k)

x = np.linspace(0, 10, 512)
plt.plot(x, list(map(lambda t: fi(1, 0, t), x)), 'b')
plt.plot(x, list(map(lambda t: fi(1, 1, t), x)), 'b')
plt.plot(x, list(map(lambda t: fi(1, 2, t), x)), 'b')
plt.plot(x, list(map(lambda t: fi(2, 0, t), x)), 'r')
plt.plot(x, list(map(lambda t: fi(2, 1, t), x)), 'r')
plt.plot(x, list(map(lambda t: fi(3, 0, t), x)), 'g')
plt.plot(x, list(map(lambda t: fi(3, 1, t), x)), 'g')
plt.plot(x, list(map(lambda t: fi(3, 2, t), x)), 'g')
plt.plot(x, list(map(lambda t: fi(3, 3, t), x)), 'g')

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



plt.show()



def describe(signal):
    w = pywt.Wavelet('haar')
    signal_with_noise = signal + np.random.uniform(-0.5, 0.5, N)

    coeffs = pywt.wavedec(signal_with_noise, w, level=6)
    print(len(signal_with_noise))

    print(signal_with_noise[165])
    sum = 0
    approx = coeffs[0]
    for ii, i in enumerate(coeffs[1:]):
        sum += i[165 // 2**(6 - ii)]
        approx = pywt.idwt(approx, i, w)
        print(6 - ii, len(i))
    print(signal_with_noise[165] - sum)
    print(np.linalg.norm(approx - signal_with_noise))

    to_show = 10
    def reconstruction_plot(yyy, color = 'r'):
        """Plot signal vector on x [0,1] independently of amount of values it contains."""
        length = len(yyy)
        plt.plot(np.linspace(0, 1, length)[:length // to_show + 1]
                 , yyy[:length // to_show + 1], color)
        # plt.plot(np.linspace(0, 1, len(yyy)), yyy, color)

    plt.plot(np.linspace(0, 1, N)[:N//to_show + 1],
             signal_with_noise[:N//to_show + 1], 'b')
    reconstruction_plot(pywt.waverec(coeffs, w), 'orange')  # full reconstruction
    # reconstruction_plot(pywt.waverec(coeffs[:-1] + [None] * 1, w)) # leaving out detail coefficients up to lvl 5
    # up_to_4 = np.array(pywt.waverec(coeffs[:-2] + [None] * 2, w))
    # up_to_3 = np.array(pywt.waverec(coeffs[:-3] + [None] * 3, w))
    # lvl_4 = up_to_4[:1000] - up_to_3[:1000]
    # reconstruction_plot(lvl_4)
    #reconstruction_plot(pywt.waverec(coeffs[:-2] + [None] * 2, w)) # leaving out detail coefficients up to lvl 4
    reconstruction_plot(pywt.waverec(coeffs[:3 + 1] + [None] * 3, w), 'r') # leaving out detail coefficients up to lvl 3
    reconstruction_plot(pywt.waverec(coeffs[:2 + 1] + [None] * 4, w), 'pink')  # leaving out detail coefficients up to lvl 2

    # reconstruction_plot(pywt.waverec(coeffs[:-5] + [None] * 5, w)) # leaving out detail coefficients up to lvl 1
    # reconstruction_plot(pywt.waverec(coeffs[:-6] + [None] * 6, w)) # leaving out all detail coefficients = reconstruction using lvl1 approximation only
    plt.show()



def sin_freq(t, freq):
    return np.sin(2 * np.pi * freq * t / frequency)
signal = np.array([sin_freq(t, 44) + 2 * sin_freq(t, 20)
                   for t in range(N)])
#describe(signal)

first_half = np.array([sin_freq(t, 44)
                       for t in range(N // 2)])
second_half = np.array([2 * sin_freq(t, 20)
                       for t in range(N // 2)])
signal = np.append(first_half, second_half)
#describe(signal)


# sin * gauss_curve = morlet
# d2f gauss_curve = mexican_hat
