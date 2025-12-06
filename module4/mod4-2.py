import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy import signal

#Steps 
#   Collect Data
#       Course Frequency Correction 
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
graph_range = 800
iq_graph_axis = [-graph_range, graph_range, -graph_range, graph_range]
numtaps = 201
cutoff_lp = 16e3
cutoff_bp = [1.2e4, 1.54e4]

# lowpass_filter = signal.firwin(numtaps, cutoff_lp, fs=fs)
bandpass_filter = signal.firwin(numtaps, cutoff_bp, fs=fs, pass_zero=False)


#Import data
# signal = np.load("/home/goose/Documents/wpi/ece-331x/module4/data0.npy")
signal = np.load("/Users/fionaprendergast/ECE331X/moredata/data2.npy")


time_array = np.arange(len(signal))/fs

#select data
signal = signal[:int(5.0 * fs)]

# Filter data 
signal = np.convolve(bandpass_filter, signal)
plt.specgram(signal, 2048, fs)
plt.show()

# plt.figure(figsize=(20, 20), num=("IQ Plots"))
# #IQ plot of raw data

# plt.subplot(2, 2, 1)
# # plt.scatter(np.real(signal), np.imag(signal), color='blue', marker='o', s=3, alpha=0.01) # Use scatter plot for points
# plt.hexbin(np.real(signal), np.imag(signal), gridsize=1000)
# plt.xlabel("Real Axis")
# plt.ylabel("Imaginary Axis")
# plt.title("IQ Plot of Raw Samples")
# plt.grid(True)
# plt.axhline(0, color='black',linewidth=0.5)
# plt.axvline(0, color='black',linewidth=0.5)
# plt.axis(iq_graph_axis) # makes the real and imaginary axes have the same scale

##Software Correct the data
#Find the center frequency 
N = len(signal)
fft_data = np.fft.fft(signal)
fft_freqs = np.fft.fftfreq(N, d=1/fs)
peak_index = np.argmax(np.abs(fft_data))
frequency_offset = fft_freqs[peak_index]
print(f"Estimated Frequency Offset: {frequency_offset} Hz")

#Correct the center frequency offset with complex exponential function (in class 11/13)
time_array = np.arange(N) / fs
correction_signal = np.exp(-1j * 2 * np.pi * frequency_offset * time_array)
signal = signal * correction_signal

# #IQ plot after course freq correction
# plt.subplot(2, 2, 2)
# # plt.scatter(np.real(signal), np.imag(signal), color='blue', marker='o', s=3, alpha=0.01) # Use scatter plot for points
# plt.hexbin(np.real(signal), np.imag(signal), gridsize=1000)
# plt.xlabel("Real Axis")
# plt.ylabel("Imaginary Axis")
# plt.title("IQ Plot of Course Frequency Corrected Samples")
# plt.grid(True)
# plt.axhline(0, color='black',linewidth=0.5)
# plt.axvline(0, color='black',linewidth=0.5)
# plt.axis(iq_graph_axis) # makes the real and imaginary axes have the same scale

# -------------------------
## Correct Phase Error, loop from pysdr.org by Dr. Marc Lichtman
# phase = 0
# freq = 0
# out = np.zeros(N, dtype=np.complex64)
# freq_log = []
# error_log = []
# phase_log=[]
# phase = 0
# freq = 0
# alpha = 0.129 
# beta = 0.00932
# for i in range(N):
#     if i % 50000 == 0:  # Progress indicator
#         print(f"Progress: {i}/{N} ({100*i/N:.1f}%)")
#     out[i] = signal[i] * np.exp(-1j*phase) # adjust the input sample by the inverse of the estimated phase offset
#     error = np.real(out[i]) * np.imag(out[i]) # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)
#     error_log.append(error)
#     # Advance the loop (recalc phase and freq offset)
#     freq += (beta * error)
#     # freq_log.append(freq * fs / (2*np.pi) / 8) # convert from angular velocity to Hz for logging
#     phase += freq + (alpha * error)

# -------------------------
# Costas Loop (fine correction) - VECTORIZED
# -------------------------
N = len(signal)
phase = np.zeros(N)
error_log = np.zeros(N)
freq = np.zeros(N)
alpha = 0.129
beta = 0.00932
out = np.zeros(N, dtype=np.complex64)

print("Running Costas Loop...")
print(f"Processing {N} samples...")

for i in range(N):
    if i % 50000 == 0:  # Progress indicator
        print(f"Progress: {i}/{N} ({100*i/N:.1f}%)")
    
    out[i] = signal[i] * np.exp(-1j*phase[i])
    error = np.real(out[i]) * np.imag(out[i])
    error_log[i] = error
    
    if i < N-1:  # Don't go out of bounds
        freq[i+1] = freq[i] + (beta * error)
        phase[i+1] = phase[i] + freq[i+1] + (alpha * error)
        
        # Phase wrapping
        phase[i+1] = np.mod(phase[i+1], 2*np.pi)

#     out[i] = signal[i] * np.exp(-1j*phase) # adjust the input sample by the inverse of the estimated phase offset
#     error = np.real(out[i]) * np.imag(out[i]) # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)
#     error_log.append(error)
#     # Advance the loop (recalc phase and freq offset)
#     freq += (beta * error)
#     # freq_log.append(freq * fs / (2*np.pi) / 8) # convert from angular velocity to Hz for logging
#     phase += freq + (alpha * error)

print("Costas Loop complete!")

#IQ plot after fine freq correction
# plt.subplot(2, 2, 3)
# # plt.scatter(np.real(out), np.imag(out), color='blue', marker='o', s=3, alpha=0.01) # Use scatter plot for points
# plt.hexbin(np.real(out), np.imag(out), gridsize=1000)
# plt.xlabel("Real Axis")
# plt.ylabel("Imaginary Axis")
# plt.title("IQ Plot of Fine and Course Frequency Corrected Samples")
# plt.grid(True)
# plt.axhline(0, color='black',linewidth=0.5)
# plt.axvline(0, color='black',linewidth=0.5)
# plt.axis(iq_graph_axis) # makes the real and imaginary axes have the same scale

# Manual shifting and filtering 
# for i in range(N):
#     if (np.angle(out[i]) > (np.arctan(187/582) + np.pi) or np.angle(out[i]) < (np.arctan(-187/582) + np.pi)) and (np.abs(out[i]) > 450): #between the angles that there is the highest density of points and is above a certian mag around pi rad
#         out[i] = np.abs(out[i])*np.exp(1j*np.pi)
#     elif(np.angle(out[i]) > (np.arctan(187/582)) or np.angle(out[i]) < (np.arctan(-187/582))) and (np.abs(out[i]) > 450): #points close to 0 rad
#         out[i] = np.abs(out[i])*np.exp(1j*0)
#     else: # otherwise discard
#         out[i] = 0 

# Rotate by 90 because of the constant phase offset
out_rot = np.zeros(N, dtype=np.complex64)
out_rot = out * np.exp(-1j * np.pi/2)

#IQ plot after everything freq correction
plt.figure(figsize=(6,6))
# plt.scatter(np.real(out_rot), np.imag(out_rot), color='blue', marker='o', s=3, alpha=0.01) # Use scatter plot for points
plt.hexbin(np.real(out), np.imag(out), gridsize=1000)
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot of Fine and Course Frequency Corrected and Filtered Samples")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.axis(iq_graph_axis) # makes the real and imaginary axes have the same scale 
plt.show()

plt.figure(figsize=(6,6))
# plt.scatter(np.real(out_rot), np.imag(out_rot), color='blue', marker='o', s=3, alpha=0.01) # Use scatter plot for points
plt.hexbin(np.real(out_rot), np.imag(out_rot), gridsize=1000)
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot of Fine and Course Frequency Corrected and Filtered Samples")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.axis(iq_graph_axis) # makes the real and imaginary axes have the same scale 
plt.show()

# plt.plot(time_array, error_log, ".-")
# plt.xlabel("Time(s)")
# plt.ylabel("Error")
# plt.title("Error vs Time of Costas Loop")
# plt.show()

# -------------------------
# Plot frequency estimate (in Hz)
# -------------------------
freq_hz = freq * fs / (2 * np.pi)

plt.figure()
plt.plot(freq_hz, '.-', markersize=2)
plt.xlabel("Sample Index")
plt.ylabel("Estimated Frequency Offset (Hz)")
plt.title("Costas Loop Frequency Estimate Over Time")
plt.grid(True)
plt.show()


# -------------------------
# Symbol timing recovery and decoding
# -------------------------
# Test different symbol rate values 
samples_per_symbol = 64  

# Simple decimation approach (for initial testing)
# Skip to middle of Costas loop output to let it settle
start_sample = len(out) // 4
symbols = out[start_sample::samples_per_symbol]

# Decode BPSK (decision: is real part positive or negative?)
bits = (np.real(symbols) > 0).astype(int)

print(f"Decoded bits: {bits}")
print(f"Number of bits: {len(bits)}")

# Try to convert to ASCII
if len(bits) % 8 == 0:
    bytes_array = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        byte_val = int(''.join(map(str, byte)), 2)
        bytes_array.append(byte_val)
    
    try:
        message = ''.join([chr(b) for b in bytes_array if 32 <= b <= 126])
        print(f"Decoded message: {message}")
    except:
        print("Could not decode to ASCII")

