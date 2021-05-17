import math
import numpy as np
from commpy.filters import rrcosfilter
from lib import *

# fc - carrier frequency
# fs - sampling rate
class Modulator:
    def __init__(self, settings):
        self.fc = settings.fc
        self.fs = settings.fs
        self.baud = settings.baud
        self.constellation = settings.constellation
        self.barker_code = settings.barker_code
        self.data = settings.data
        _, self.rrc = rrcosfilter(N=int(12 * self.fs / self.baud), alpha=0.8, Ts=self.fs / self.baud, Fs=1)

    # Convert text to array of ints.
    def _str2ints(self, text):
        return [42 if (ord(x) < 32 or ord(x) > 126) else ord(x) for x in text]  # filter unused symbols

    # Use gray code of size n to encode int.
    def _encode_gray_code(self, x):
        x = x ^ (x >> 1)
        return "{0:0{1}b}".format(x, 8)

    # Encode byte into array of four constellation points.
    def _encode_int(self, x):
        x = self._encode_gray_code(x)
        return np.array([self.constellation[x[6:8]], self.constellation[x[4:6]],
                self.constellation[x[2:4]], self.constellation[x[0:2]]])

    # Encode all data into constellation points.
    def _encode_constellation(self):
        # code symbols from alphabet {1, -1, j, -j}
        self.data = np.concatenate(list(map(self._encode_int, self.data)))

    # Increase discretion of data up to frequency fd (in KHz) by adding zeros between points.
    def _increase_discretion(self):
        num_add = round(self.fs / 1000) - 1
        values = np.zeros(num_add)
        increased = np.array([], dtype=complex)
        for i in range(self.data.size):
            if i == 0:
                increased = np.concatenate(([self.data[i]], values))
            else:
                increased = np.concatenate((increased, [self.data[i]], values))
        self.data = increased

    # Encode string: text is converted to ints, encoded with gray code and passed to constellation (+increased discretion).
    def _encode_string(self, text):
        self.data = self._str2ints(text)
        self._encode_constellation()
        self.data = np.concatenate((self.barker_code, self.data)) # add Barker code
        self._increase_discretion()
        return self.data

    def _add_zeros(self):
        zeros = np.zeros(int(10*self.fs/self.baud))
        self.data = np.concatenate((zeros, self.data, zeros))

    def _apply_filter(self):
        self.data = signal.fftconvolve(self.data, self.rrc, mode='same') # self.rrc[np.newaxis:]
        self.data /= np.max(self.data) # needed?
        # t_rrc = np.arange(len(self.rrc)) / self.fs  # the time points that correspond to the filter values
        # fig, ax = plt.subplots()
        # ax.set_title("Filter")
        # ax.plot(t_rrc/1e-3, self.rrc)

    # Modulate signal over carrier frequency (self.fc)
    def _modulate_signal(self):
        carrier = 2 * math.pi * self.fc / self.fs * np.arange(0, len(self.data))
        self.data = (self.data * np.exp(1j*carrier)).real
        return self.data

    def transmit_data(self, text, show_src=False, show_filt=False, show_mod=False):
        self._encode_string(text)
        self._add_zeros()
        if (show_src): plot_spectrum(self.data, self.fs, "Source", "stem")
        self._apply_filter()
        if (show_filt): plot_spectrum(self.data, self.fs, "Filtered")
        self._modulate_signal()
        if (show_mod): plot_spectrum(self.data, self.fs, "Modulated")
        return self.data
