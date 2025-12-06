from scipy import signal
import numpy as np

#Lowpass filter
def lowpass(cutoff, taps, fs, data):
    lowpass_filter = signal.firwin(taps, cutoff, fs=fs)
    filtered_data = np.convolve(lowpass_filter, data)
    return filtered_data

def phaseDiff(data):
    phase = np.angle(data)
    phase_unwrap = np.unwrap(phase)
    phase_diff = np.diff(phase_unwrap)
    return phase_diff

