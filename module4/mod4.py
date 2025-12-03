import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#Steps 
#   Collect Data
#       coarse  Frequency Correction 
#       Make sure gain is good
#   Implement DPLL
#       Basically a pid loop
#       Graph phase error over time 
#

#Variables 

fc = 915e6 #center freq
fs = 521e3 #Sampling rate
bufferSize = 2**16 #Buffer size
runTime = 30 #Run time
bandwidth = 26e4 #Bandwidth 
samples = runTime*fs/bufferSize #number of sample buffers
graph_range = 300
iq_graph_axis = [-graph_range, graph_range, -graph_range, graph_range]
numtaps = 201
cutoff_lp = 16e3
cutoff_bp = [1.2e4, 1.54e4]

# lowpass_filter = signal.firwin(numtaps, cutoff_lp, fs=fs)
bandpass_filter = signal.firwin(numtaps, cutoff_bp, fs=fs, pass_zero=False)


#Import data
signal = np.load("/home/goose/Documents/wpi/ece-331x/module4/data0.npy")


time_array = np.arange(len(signal))/fs

#select data
signal = signal[:int(5*fs)]

# Filter data 
signal = np.convolve(bandpass_filter, signal)

plt.figure(figsize=(20, 20), num=("IQ Plots"))
#IQ plot of raw data

plt.subplot(2, 2, 1)
# plt.scatter(np.real(signal), np.imag(signal), color='blue', marker='o', s=3, alpha=0.01) # Use scatter plot for points
plt.hexbin(np.real(signal), np.imag(signal), gridsize=1000)
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot of Raw Samples")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.axis(iq_graph_axis) # makes the real and imaginary axes have the same scale

##Software Correct the data
#Coarse Frequency Correction-Fiona 
N = len(signal)
signal_squared = signal ** 2
psd = np.fft.fftshift(np.abs(np.fft.fft(signal_squared)))
f = np.linspace(-fs / 2.0, fs / 2.0, len(psd), endpoint=False)
max_freq = f[np.argmax(psd)]
coarse_offset = max_freq / 2.0
print(f"Estimated frequency offset (coarse, Hz): {coarse_offset}")

Ts = 1.0 / fs
t = np.arange(len(signal)) * Ts

# Correct original (unsquared) signal
signal = signal * np.exp(-1j * 2 * np.pi * coarse_offset * t / 2.0)


#IQ plot after coarse  freq correction
plt.subplot(2, 2, 2)
# plt.scatter(np.real(signal), np.imag(signal), color='blue', marker='o', s=3, alpha=0.01) # Use scatter plot for points
plt.hexbin(np.real(signal), np.imag(signal), gridsize=1000)
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot of coarse  Frequency Corrected Samples")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.axis(iq_graph_axis) # makes the real and imaginary axes have the same scale


## Correct Phase Error, loop from pysdr.org by Dr. Marc Lichtman
phase = 0
freq = 0
out = np.zeros(N, dtype=np.complex64)
freq_log = []
error_log = []
phase_log=[]
phase = 0
freq = 0 #0.153, 0.00928
alpha = 0.145 #0.129, 0.145 
beta = 0.00932 #0.00932
for i in range(N):
    out[i] = signal[i] * np.exp(-1j*phase) # adjust the input sample by the inverse of the estimated phase offset
    error = np.real(out[i]) * np.imag(out[i]) # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)
    error_log.append(error)
    # Advance the loop (recalc phase and freq offset)
    freq += (beta * error)
    # freq_log.append(freq * fs / (2*np.pi) / 8) # convert from angular velocity to Hz for logging
    phase += freq + (alpha * error)

out = out * np.exp(-1j*np.pi/2)

np.save("/home/goose/Documents/wpi/ece-331x/module4/filtered_data.npy", out)
#IQ plot after fine freq correction
plt.subplot(2, 2, 3)
# plt.scatter(np.real(out), np.imag(out), color='blue', marker='o', s=3, alpha=0.01) # Use scatter plot for points
plt.hexbin(np.real(out), np.imag(out), gridsize=1000)
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot of Fine and coarse  Frequency Corrected Samples")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.axis(iq_graph_axis) # makes the real and imaginary axes have the same scale

plt.show()

time_array = np.arange(len(error_log))/fs

plt.plot(time_array, error_log)
plt.xlabel("Time(s)")
plt.ylabel("Error")
plt.title("Error vs Time of Costas Loop")

plt.show()
