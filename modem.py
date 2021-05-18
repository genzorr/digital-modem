import sys
import numpy as np
import matplotlib.pyplot as plt
import difflib
import lib
from modulator import *
from demodulator import *
from channel import *

message = 'lfhsdkasdbabsfksdj'

if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)

    settings = lib.Settings(fc=7000, fs=48000)
    modulator = Modulator(settings=settings)
    modulated_signal = modulator.transmit_data(message, show_src=False, show_filt=True, show_mod=False)

    channel = Channel()
    signal = channel.send(modulated_signal)
    # plot_spectrum(signal, 48000, "Channel")

    demodulator = Demodulator(settings=settings)
    demodulator.receive_data(signal)
    result = demodulator.process_data(show_demod=True, show_filt=True, show_res=False, show_2d=True)

    print("Sent: ", message)
    print("Recv: ", result)
    diff = [li for li in difflib.ndiff(message, result) if li[0] != ' ']
    print("Diff: ", diff)

    plt.show()
