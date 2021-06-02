import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
from commpy.filters import rrcosfilter
import scipy.signal as signal
import scipy.integrate as integrate
from lib import *

''' 
signal = I * cos(2pi*f_c*t) - Q * sin(2pi*f_c*t)
'''

class Demodulator:
    def __init__(self, settings):
        self.fc = settings.fc
        self.fs = settings.fs
        self.baud = settings.baud
        self.constellation = {v: k for k, v in settings.constellation.items()}
        self.barker_code = settings.barker_code
        self.data = settings.data
        self.sps = round(self.fs / self.baud)
        _, self.rrc = rrcosfilter(N=int(12 * self.sps), alpha=0.35, Ts=self.sps, Fs=1)
        self.rrc = self.rrc / max(abs(self.rrc))
        # self.rrc = Filter(b=self.rrc / max(abs(self.rrc)))

    def receive_data(self, data):
        self.data = data

    def _demodulate_signal(self):
        carrier = 2 * math.pi * self.fc / self.fs * np.arange(0, len(self.data))
        self.data = self.data * np.exp(-1j*carrier)

    def _apply_filter(self):
        # self.data = self.rrc(self.data)
        # self.data /= np.max(abs(self.data))
        self.data = signal.fftconvolve(self.data, self.rrc, mode='same')
        self.data /= np.max(np.abs(self.data))
        plot_spectrum(self.data, self.fs, "Pre-divided")
        value = np.argmax(np.abs(self.data))
        self.data /= self.data[value]
        # t_rrc = np.arange(len(self.rrc)) / self.fs  # the time points that correspond to the filter values
        # fig, ax = plt.subplots()
        # ax.set_title("Filter")
        # ax.plot(t_rrc/1e-3, self.rrc)

    def _decrease_discretion(self):
        N = round(self.sps)
        self.data = self.data[0::N]

    def _find_barker(self):
        # up barker
        Bar = np.array(UpSample(self.barker_code, self.sps))

        fig, ax = plt.subplots()
        samples = range(len(self.data))
        ax.plot(samples, self.data, label='signal')

        # correlation
        autocorr_filt = Filter(a=[1], b=np.array([i for i in reversed(Bar)], dtype='complex'))
        correlation = autocorr_filt(self.data)
        ax.plot(samples, correlation, label='correlation')

        # power
        # power = np.array([x * x for x in self.data])
        power = np.abs(self.data)
        ax.plot(samples, power, label='power')

        # average power????
        rect_filt = Filter(a=[1], b=np.ones(len(self.barker_code) * self.sps) / (len(self.barker_code) * self.sps))
        rect_filt_power = rect_filt(abs(power))
        ax.plot(samples, np.sqrt(rect_filt_power), label='average power')

        ax.grid()
        ax.legend()

        index = abs(correlation).argmax()
        if correlation[index] > 3 * np.sqrt(abs(rect_filt_power))[index] * math.sqrt(len(self.barker_code)):
            return index
        else:
            print("Error! Cannot find the begining!")
            return 0

    def _remove_barker(self):
        start = self._find_barker() + 1
        self.data = self.data[start:]
        # self.data /= np.max(self.data)

    def _plot_data_2d(self):
        fig, ax = plt.subplots()
        for p in self.data:
            ax.scatter(p.real, p.imag, s=5)
        ax.set_ylabel('Q')
        ax.set_xlabel('I')
        ax.grid(b=True, which='major', color='k', alpha=0.1)
        ax.grid(axis='y', which='minor', color='k', linestyle=':', alpha=0.1)

    # Convert array of ints to text.
    def _ints2str(self, ints):
        return ''.join(list(map(chr, ints)))

    # Use gray code of size n to decode int.
    def _decode_gray_code(self, x):
        y = x
        while x:
            x = x >> 1
            y ^= x
        return y

    # Decode points from alphabet {1, -1, j, -j}
    def _decode_point(self, p):
        x, y = p.real, p.imag
        border = 0.25
        if (p.real > border): x = 1
        elif (p.real < -border): x = -1
        else: x = 0

        if (p.imag > border): y = 1
        elif (p.imag < -border): y = -1
        else: y = 0
        p = x + 1j*y
        return self.constellation.get(p)

    # Decode all data from constellation points.
    def _decode_constellation(self):
        self.data = list(map(self._decode_point, self.data))
        self.data = ''.join(['?' if x is None else x for x in self.data])

    def _decode_string(self):
        self._decode_constellation()
        text = []
        for i in range(0, len(self.data), 8):
            symb = self.data[i+6:i+8]+self.data[i+4:i+6]+self.data[i+2:i+4]+self.data[i:i+2]
            symb = self._decode_gray_code(int(symb,2)) if '?' not in symb else 126
            text.append(symb)
        self.data = self._ints2str(text)

    def process_data(self, show_demod=False, show_filt=False, show_res=False, show_2d=False):
        self._demodulate_signal()
        if (show_demod): plot_spectrum(self.data, self.fs, "Demodulated")
        self._apply_filter()
        if (show_filt): plot_spectrum(self.data, self.fs, "Filtered")
        self._remove_barker()
        self._decrease_discretion()
        if (show_res): plot_spectrum(self.data, self.fs, "Result", "stem")
        if (show_2d): self._plot_data_2d()

        self._decode_string()
        return self.data