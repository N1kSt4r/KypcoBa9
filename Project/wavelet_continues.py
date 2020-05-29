import numpy as np
import matplotlib.pyplot as plt
import pywt

wavelet = pywt.ContinuousWavelet('mexh')
plt.plot(wavelet.wavefun()[1] * 2, wavelet.wavefun()[0])
plt.plot(wavelet.wavefun()[1], wavelet.wavefun()[0])
plt.show()

frequency = 1000
# в сигнале частоты от 0 до 500
N = 1000
# количество данных, 1 секунды для такой частоты

def describe(signal):
    w = pywt.ContinuousWavelet('mexh')
    signal_with_noise = signal + np.random.uniform(-0.5, 0.5, N)

    amp, scales = pywt.cwt(signal_with_noise,
                           2**np.arange(7),
                           w, sampling_period=0.25)
    to_show = 5
    def reconstruction_plot(yyy, color = 'r'):
        """Plot signal vector on x [0,1] independently of amount of values it contains."""
        length = len(yyy)
        mean = length // 2
        delta = length // to_show // 2
        plt.plot(np.linspace(0, 1, length)[mean - delta:mean + delta + 1]
                 , yyy[mean - delta:mean + delta + 1], color)
        # plt.plot(np.linspace(0, 1, len(yyy)), yyy, color)

    # plt.plot(np.linspace(0, 1, N)[:N//to_show + 1],
    #          signal_with_noise[:N//to_show + 1], 'b')
    reconstruction_plot(signal_with_noise, 'orange')
    reconstruction_plot(amp[0], 'r')
    reconstruction_plot(amp[2], 'g')
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


# sin * gauss_curve = morlet
# d2f gauss_curve = mexican_hat
