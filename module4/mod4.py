import numpy as np
import matplotlib.pyplot as plt

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
signal = np.load("/home/goose/Documents/wpi/ece-331x/module4/data3.npy")

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
course_corr_signal = signal * correction_signal

## Correct Phase Error
fine_corr_signal = []
# Detect phase and send it to 0 or 180 
for point in course_corr_signal:
    if np.abs(point) < 100:
        fine_corr_signal.append(0)
        continue
    if (np.pi/2 < np.angle(point) or -1*np.pi/2 > np.angle(point)):
        corr_angle = np.pi
    else:
        corr_angle = 0
    fine_corr_signal.append(np.abs(point)*np.exp(1j * corr_angle))



# plt.scatter(np.real(fine_corr_signal), np.imag(fine_corr_signal), color='blue', marker='o', s=3, alpha=0.4) # Use scatter plot for points
# plt.xlabel("Real Axis")
# plt.ylabel("Imaginary Axis")
# plt.title("IQ Plot After Frequency Correction")
# plt.grid(True)
# plt.axhline(0, color='black',linewidth=0.5)
# plt.axvline(0, color='black',linewidth=0.5)
# plt.axis('equal') # makes the real and imaginary axes have the same scale
plt.plot(time_array, np.angle(fine_corr_signal))
plt.title('Signal Phase vs. Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Phase (radians)')
plt.grid(True)  

plt.show()
