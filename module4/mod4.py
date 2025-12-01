import numpy as np
import matplotlib.pyplot as plt
import imageio
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
iq_graph_axis = [-1000,1000,1000,-1000]

#Import data
signal = np.load("/home/goose/Documents/wpi/ece-331x/module4/data0.npy")

time_array = np.arange(len(signal))/fs

#select data
signal = signal[:int(5*fs)]

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

#IQ plot after course freq correction
plt.subplot(2, 2, 2)
# plt.scatter(np.real(signal), np.imag(signal), color='blue', marker='o', s=3, alpha=0.01) # Use scatter plot for points
plt.hexbin(np.real(signal), np.imag(signal), gridsize=1000)
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot of Course Frequency Corrected Samples")
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
freq = 0
alpha = 0.129 
beta = 0.00932
for i in range(N):
    out[i] = signal[i] * np.exp(-1j*phase) # adjust the input sample by the inverse of the estimated phase offset
    error = np.real(out[i]) * np.imag(out[i]) # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)
    error_log.append(error)
    # Advance the loop (recalc phase and freq offset)
    freq += (beta * error)
    # freq_log.append(freq * fs / (2*np.pi) / 8) # convert from angular velocity to Hz for logging
    phase += freq + (alpha * error)


#IQ plot after fine freq correction
plt.subplot(2, 2, 3)
# plt.scatter(np.real(out), np.imag(out), color='blue', marker='o', s=3, alpha=0.01) # Use scatter plot for points
plt.hexbin(np.real(out), np.imag(out), gridsize=1000)
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot of Fine and Course Frequency Corrected Samples")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.axis(iq_graph_axis) # makes the real and imaginary axes have the same scale

# Manual shifting and filtering 
for i in range(N):
    if (np.angle(out[i]) > (np.arctan(187/582) + np.pi) or np.angle(out[i]) < (np.arctan(-187/582) + np.pi)) and (np.abs(out[i]) > 450): #between the angles that there is the highest density of points and is above a certian mag around pi rad
        out[i] = np.abs(out[i])*np.exp(1j*np.pi)
    elif(np.angle(out[i]) > (np.arctan(187/582)) or np.angle(out[i]) < (np.arctan(-187/582))) and (np.abs(out[i]) > 450): #points close to 0 rad
        out[i] = np.abs(out[i])*np.exp(1j*0)
    else: # otherwise discard
        out[i] = 0 




#IQ plot after everything freq correction
plt.subplot(2, 2, 4)
plt.scatter(np.real(out), np.imag(out), color='blue', marker='o', s=3, alpha=0.01) # Use scatter plot for points
# plt.hexbin(np.real(out), np.imag(out), gridsize=1000)
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot of Fine and Course Frequency Corrected and Filtered Samples")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.axis(iq_graph_axis) # makes the real and imaginary axes have the same scale 
plt.show()

plt.plot(time_array, error_log, ".-")
plt.xlabel("Time(s)")
plt.ylabel("Error")
plt.title("Error vs Time of Costas Loop")
plt.show()