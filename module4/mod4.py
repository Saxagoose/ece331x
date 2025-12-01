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
alpha = 0.132
beta = 0.00932
out = np.zeros(N, dtype=np.complex64)
freq_log = []
filenames = []
ii = 0
iii = 0
error_log = []
for i in range(N):
    out[i] = signal[i] * np.exp(-1j*phase) # adjust the input sample by the inverse of the estimated phase offset
    error = np.real(out[i]) * np.imag(out[i]) # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)
    error_log.append(error)
    # Advance the loop (recalc phase and freq offset)
    freq += (beta * error)
    # freq_log.append(freq * fs / (2*np.pi) / 8) # convert from angular velocity to Hz for logging
    phase += freq + (alpha * error)
    # Optional: Adjust phase so its always between 0 and 2pi, recall that phase wraps around every 2pi
    # while phase >= 2*np.pi:
    #     phase -= 2*np.pi
    # while phase < 0:
    #     phase += 2*np.pi
    # if ii % 1000 == 0:
    #     # Plot animation frame
    #     fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 7))
    #     fig.subplots_adjust(hspace=0.4)
    #     ax1.plot(freq_log, '.-')
    #     ax1.set_xlabel('Sample')
    #     ax1.set_ylabel('Freq Offset [Hz]')
    #     ax2.plot(np.real(out[(i-20):i]), np.imag(out[(i-20):i]), '.')
    #     ax2.axis([-2, 2, -0.8, 0.8])
    #     ax2.set_ylabel('Q')
    #     ax2.set_xlabel('I')
    #     #plt.show()
    #     filename = '/tmp/costas_' + str(iii) + '.png'
    #     iii += 1
    #     print(iii)
    #     fig.savefig(filename, bbox_inches='tight')
    #     filenames.append(filename)
    #     plt.close(fig)

# Plot freq over time to see how long it takes to hit the right offset
# images = []
# for filename in filenames:
#     images.append(imageio.imread(filename))
# imageio.mimsave('/tmp/costas.gif', images, fps=20)
plt.plot(error_log, ".-")
plt.show()


# plt.scatter(np.real(out), np.imag(out), color='blue', marker='o', s=3, alpha=0.01) # Use scatter plot for points
plt.hexbin(np.real(signal), np.imag(signal), gridsize=200)
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
