import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.fft as fft

class Filter(object):
    def __init__(self, b, a = [1], init=[]):
        self.b = b
        self.a = a
        self.clear(init)
    def __call__(self, x):
        y, self.state = signal.lfilter(self.b, self.a, x, zi=self.state)
        return y
    def clear(self, init=[]):
        self.state = signal.lfiltic(self.b, self.a, [], init)

def UpSample(msg, SPS):
	res = []
	for s in msg:
		res += [s] + [0] * (SPS-1)
	return res

class Settings:
    def __init__(self, fc, fs):
        self.fc = fc
        self.fs = fs
        self.baud = 1000
        self.constellation = {'00': 1, '11': -1, '10': -1j, '01': 1j}
        self.barker_code = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=complex)
        self.data = None

def plot_signal(data, text="", param=""):
    fig, ax = plt.subplots()
    ax.set_title(text + " signal")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in')
    ax.grid(b=True, which='major', color='k', alpha=0.1)
    ax.grid(axis='y', which='minor', color='k', linestyle=':', alpha=0.1)

    N = len(data)
    if param == "":
        ax.plot(np.arange(0, N, 1), data.real, lw=1)
        ax.plot(np.arange(0, N, 1), data.imag, lw=1)
    else:
        ax.stem(np.arange(0, N, 1), data.real, linefmt='tab:blue', markerfmt=' ')
        ax.stem(np.arange(0, N, 1), data.imag, linefmt='tab:orange', markerfmt=' ')

def plot_spectrum(data, fs, text="", param=""):
    fig, ax = plt.subplots(2)
    ax[0].set_title(text + " signal")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("x")
    ax[0].minorticks_on()
    ax[0].tick_params(axis='both', which='both', direction='in')
    ax[0].grid(b=True, which='major', color='k', alpha=0.1)
    ax[0].grid(axis='y', which='minor', color='k', linestyle=':', alpha=0.1)

    ax[1].set_title(text + " spectrum")
    ax[1].set_xlabel("f")
    ax[1].set_ylabel("X")
    ax[1].minorticks_on()
    ax[1].tick_params(axis='both', which='both', direction='in')
    ax[1].grid(b=True, which='major', color='k', alpha=0.1)
    ax[1].grid(axis='y', which='minor', color='k', linestyle=':', alpha=0.1)

    # Apply rrc window before fft.
    N = len(data)
    rrc = (1 + np.cos(math.pi * (2*np.arange(0,N,1)/N + 1)))/2
    yf = fft.fft(data*rrc)
    xf = fft.fftfreq(N, 1 / fs)[:N // 2]
    yf = 2.0 / N * np.abs(yf[0:N // 2])

    if param == "":
        ax[0].plot(np.arange(0, N, 1), data.real, lw=1)
        ax[0].plot(np.arange(0, N, 1), data.imag, lw=1)
    else:
        ax[0].stem(np.arange(0, N, 1), data.real, linefmt='tab:blue', markerfmt=' ')
        ax[0].stem(np.arange(0, N, 1), data.imag, linefmt='tab:orange', markerfmt=' ')
    ax[1].plot(xf, yf, lw=1)
    ax[1].set_yscale('log')
