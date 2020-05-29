import numpy as np
import matplotlib.pyplot as plt

frequency = 1000
# в сигнале частоты от 0 до 100
N = 1000
# количество данных, 1 секунды для такой частоты

def describe(signal):
    signal_with_noise = signal + np.random.uniform(-2, 2, N)
    spectr = np.fft.rfft(signal_with_noise)

    plt.plot(np.arange(N) / frequency, signal_with_noise)
    plt.plot(np.arange(N) / frequency, signal, 'r')
    plt.show()

    #print(np.fft.rfftfreq(N, 1./frequency)[np.abs(spectr) / N > 0.07])
    #print(np.abs(spectr[np.abs(spectr) / N > 0.07]) / N)
    plt.plot(np.fft.rfftfreq(N, 1./frequency), np.abs(spectr)/N)
    plt.show()


def sin_freq(t, freq):
    return np.sin(2 * np.pi * freq * t / frequency)
signal = np.array([sin_freq(t, 44) + 2 * sin_freq(t, 20)
                   for t in range(N)])
describe(signal)

first_half = np.array([sin_freq(t, 44)
                       for t in range(N // 2)])
second_half = np.array([2 * sin_freq(t, 20)
                       for t in range(N // 2)])
signal = np.append(first_half, second_half)
describe(signal)

'''
import pywt
w = pywt.Wavelet('sym5')
#plt.plot(w.dec_lo)
coeffs = pywt.wavedec(signal_with_noise, w, level=6)

plt.plot(np.linspace(0, 1, len(coeffs[0])), coeffs[0])
plt.plot(np.linspace(0, 1, len(coeffs[2])), coeffs[2])
plt.plot(np.linspace(0, 1, len(coeffs[4])), coeffs[4])
plt.show()
def reconstruction_plot(yyy, **kwargs):
    """Plot signal vector on x [0,1] independently of amount of values it contains."""
    plt.plot(np.linspace(0, 1, len(yyy)), yyy, **kwargs)

reconstruction_plot(pywt.waverec(coeffs, w)) # full reconstruction
#reconstruction_plot(pywt.waverec(coeffs[:-1] + [None] * 1, w)) # leaving out detail coefficients up to lvl 5
#reconstruction_plot(pywt.waverec(coeffs[:-2] + [None] * 2, w)) # leaving out detail coefficients up to lvl 4
#reconstruction_plot(pywt.waverec(coeffs[:-3] + [None] * 3, w)) # leaving out detail coefficients up to lvl 3
reconstruction_plot(pywt.waverec(coeffs[:-4] + [None] * 4, w)) # leaving out detail coefficients up to lvl 2
#reconstruction_plot(pywt.waverec(coeffs[:-5] + [None] * 5, w)) # leaving out detail coefficients up to lvl 1
#reconstruction_plot(pywt.waverec(coeffs[:-6] + [None] * 6, w)) # leaving out all detail coefficients = reconstruction using lvl1 approximation only
plt.legend(['Full reconstruction', 'Reconstruction using detail coefficients lvl 1+2', 'Reconstruction using lvl 1 approximation only'])
plt.show()'''

# sin * gauss_curve = morlet
# d2f gauss_curve = mexican_hat
