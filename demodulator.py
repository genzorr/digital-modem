import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
from commpy.filters import rrcosfilter
import scipy.signal as signal
import scipy.integrate as integrate
from lib import *

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
        _, self.rrc = rrcosfilter(N=int(12 * self.fs / self.baud), alpha=0.8, Ts=self.fs / self.baud, Fs=1)

    def receive_data(self, data):
        self.data = data

    def _demodulate_signal(self):
        carrier = 2 * math.pi * self.fc / self.fs * np.arange(0, len(self.data))
        self.data = self.data * np.exp(-1j*carrier)

        # N = round(self.fs/self.baud)
        # s = []
        # l = len(self.data)
        # for k in range(0, l, N):
        #     s.append(integrate.simps(self.data[k:k+N]))
        # self.data = np.array(s)

    def _apply_filter(self):
        self.data = signal.fftconvolve(self.data, self.rrc, mode='same')
        self.data /= np.max(self.data) # needed?
        # t_rrc = np.arange(len(rrc)) / self.fs  # the time points that correspond to the filter values
        # fig, ax = plt.subplots()
        # ax.set_title("Filter")
        # ax.plot(t_rrc/1e-3, rrc)

    def _decrease_discretion(self):
        N = round(self.fs / self.baud)
        self.data = self.data[0::N]

    def _find_barker(self):
        # up barker
        SPS = round(self.fs / self.baud)
        Bar = np.array(UpSample(self.barker_code, SPS))

        plt.figure()
        samples = range(len(self.data))
        plt.plot(samples, self.data, label='signal')

        # correlation
        autocorr_filt = Filter(a=[1], b=np.array([i for i in reversed(Bar)], dtype='complex'))
        correlation = autocorr_filt(self.data)
        plt.plot(samples, correlation, label='correlation result')

        # power
        power = np.array([x * x for x in self.data])
        plt.plot(samples, power, label='power')

        # average power????
        rect_filt = Filter(a=[1], b=np.ones(len(self.barker_code) * SPS) / (len(self.barker_code) * SPS))
        rect_filt_power = rect_filt(abs(power))
        plt.plot(samples, np.sqrt(rect_filt_power), label='sqrt(rect_flt(power))')

        plt.grid()
        plt.legend()
        plt.show()
        plt.close()

        index = abs(correlation).argmax()
        if correlation[index] > 3 * np.sqrt(abs(rect_filt_power))[index] * math.sqrt(len(self.barker_code)):
            return index
        else:
            print("Error! Cannot find the begining!")

    def _remove_barker(self):
        # correlator = Filter(a=[1], b=np.flip(self.barker_code))
        # res = correlator(self.data)
        # plot_signal(res, "Correlator", "stem")
        # start = np.argmax(res) + 1
        start = self._find_barker() + 1
        self.data = self.data[start:]

    def _plot_data_2d(self):
        fig, ax = plt.subplots()
        ax.scatter(self.data.real, self.data.imag, s=5)
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
        if (p.real > 0.5): x = 1
        elif (p.real < -0.5): x = -1
        else: x = 0

        if (p.imag > 0.5): y = 1
        elif (p.imag < -0.5): y = -1
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