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

#Import data
signal = np.load("/home/goose/Documents/wpi/ece-331x/module4/data0.npy")

time_array = np.arange(len(signal))/fs

#select data
signal = signal[:int(5*fs)]



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

# plt.specgram(signal, NFFT=2048, Fs=fs)
# plt.show()
# Filter out some noise 
# cutoff_magnitude = 55
# for x in range(len(signal)):
#     if np.abs(signal[x]) < cutoff_magnitude:
#         signal[x] = 0

## Correct Phase Error
phase = 0
freq = 0
# These next two params is what to adjust, to make the feedback loop faster or slower (which impacts stability)

out = np.zeros(N, dtype=np.complex64)
freq_log = []
filenames = []
ii = 0
iii = 0
error_log = []
phase_log=[]
# for x in np.linspace(0, 1, ):
phase = 0
freq = 0
alpha = 0.129 #0.129
beta = 0.00932
for i in range(N):
    out[i] = signal[i] * np.exp(-1j*phase) # adjust the input sample by the inverse of the estimated phase offset
    error = np.real(out[i]) * np.imag(out[i]) # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)
    error_log.append(error)
    # Advance the loop (recalc phase and freq offset)
    freq += (beta * error)
    freq_log.append(freq * fs / (2*np.pi) / 8) # convert from angular velocity to Hz for logging
    phase += freq + (alpha * error)

    # plt.hexbin(np.real(out), np.imag(out), gridsize=200)
    # plt.xlabel("Real Axis")
    # plt.ylabel("Imaginary Axis")
    # plt.title(f"Alpha: {alpha}, Beta: {beta}")
    # plt.grid(True)
    # plt.axhline(0, color='black',linewidth=0.5)
    # plt.axvline(0, color='black',linewidth=0.5)
    # plt.axis([-1000,1000,1000,-1000]) # makes the real and imaginary axes have the same scale
    # filename = '/tmp/costa/costas_' + str(iii) + '.png'
    # plt.savefig(filename)
    # print(iii)
    # iii += 1




# plt.plot(error_log, ".-")
# plt.show()


# plt.scatter(np.real(out), np.imag(out), color='blue', marker='o', s=3, alpha=0.01) # Use scatter plot for points
plt.hexbin(np.real(out), np.imag(out), gridsize=200)
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot After Frequency Correction")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
# plt.xlim(-25, 25)
# plt.ylim(-25, 25)
plt.axis([-1000,1000,1000,-1000]) # makes the real and imaginary axes have the same scale
# plt.plot(time_array, np.angle(fine_corr_signal))
# plt.title('Signal Phase vs. Time')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Phase (radians)')
# plt.grid(True)  

plt.show()
