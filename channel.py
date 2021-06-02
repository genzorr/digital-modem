import numpy as np

class Channel:
    def __init__(self):
        self.rng = np.random.default_rng()

    def send(self, signal):
        random_delay = np.random.randint(200, 2000)
        signal = np.append(np.zeros(random_delay), signal)
        # return signal
        noise = self.rng.normal(0, 0.5, len(signal))
        print(f'Channel: delay = {random_delay}, noise = {np.max(noise)}')
        return signal + noise
