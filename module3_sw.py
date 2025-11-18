import numpy as np
import matplotlib.pyplot as plt

# Load data from module 1
signal = np.load("/Users/fionaprendergast/ECE331X/data.npy") # Change path for your computer
hw_corrected_signal = np.load("/Users/fionaprendergast/ECE331X/data1.npy") # Change path for your computer

# Sample rate from module 1
Fs = 512e3  # Sampling frequency from module 1

# Get the useful data from module 1 9.12-9.91 seconds
signal = signal[int(9.12*Fs):int(9.91*Fs)] # Timing for old not corrected data
hw_corrected_signal = hw_corrected_signal[int(13.00*Fs):int(13.79*Fs)] # Timing for HW corrected data

#Creates spectrogram (module 1)
# plt.specgram(signal, Fs=Fs)
# plt.xlabel("Time(s)")
# plt.ylabel("Frequency(Hz)")
# plt.show()

# Calculate frequency offset using FFT
N = len(signal)
fft_data = np.fft.fft(signal)
fft_freqs = np.fft.fftfreq(N, d=1/Fs)
peak_index = np.argmax(np.abs(fft_data))
frequency_offset = fft_freqs[peak_index]
print(f"Estimated Frequency Offset: {frequency_offset} Hz")

frequency_offset = 3.8e4 # Eyeballed frequency offset value 

# Correct frequency offset with complex exponential function (in class 11/13)
time_array = np.arange(N) / Fs
correction_signal = np.exp(-1j * 2 * np.pi * frequency_offset * time_array)
corrected_signal = signal * correction_signal

corrected_signal = hw_corrected_signal

# Calculate the data arrays to plot
magnitude_array = np.abs(signal) #Takes the magnitude of the complex numbers in the data set
time_array = np.arange(len(signal)) / Fs
cor_magnitude_array = np.abs(corrected_signal)

phase_array = np.unwrap(np.angle(signal)) #Takes the phase of the complex numbers in the data set
cor_phase_array = np.unwrap(np.angle(corrected_signal)) #Takes the phase of the complex numbers in the data set
# phase_array = np.angle(signal) #Takes the phase of the complex numbers in the data set
# cor_phase_array = np.angle(corrected_signal) #Takes the phase of the complex numbers in the data set

# Also plot the IQ constellation before and after correction
x_coords = signal.real
y_coords = signal.imag
corrected_x_coords = corrected_signal.real
corrected_y_coords = corrected_signal.imag

# Plot signal magnitude vs. time before and after correction
plt.figure(figsize=(30, 6), num=("Magnitude Plots"))
plt.subplot(1, 3, 1)
plt.plot(time_array, magnitude_array)
plt.title('Signal Magnitude vs. Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(time_array, cor_magnitude_array)
plt.title('Corrected Signal Magnitude vs. Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Magnitude')
plt.grid(True)

# To visualize, plot the magnitude data on top of each other in one plot
plt.subplot(1,3,3)
plt.plot(time_array, magnitude_array, label='Original Signal', alpha=0.7)
plt.plot(time_array, cor_magnitude_array, label='Corrected Signal', alpha=0.7)
plt.title('Signal Magnitude vs. Time (Before and After Correction)')
plt.xlabel('Time (seconds)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
# plt.savefig("magnitude.png")
plt.show()

# Plot phase vs. time before and after correction
plt.figure(figsize=(30, 6), num=("Phase Plots"))
plt.subplot(1, 3, 1)
plt.plot(time_array, phase_array)
plt.title('Signal Phase vs. Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Phase (radians)')
plt.grid(True)  

plt.subplot(1, 3, 2)
plt.plot(time_array, cor_phase_array)
plt.title('Corrected Signal Phase vs. Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Phase (radians)')
plt.grid(True)  

# To vizualize, plot the phase data on top of each other in one plot
plt.subplot(1,3,3)
plt.plot(time_array, phase_array, label='Original Signal', alpha=0.7)
plt.plot(time_array, cor_phase_array, label='Corrected Signal', alpha=0.7)
plt.title('Signal Phase vs. Time (Before and After Correction)')
plt.xlabel('Time (seconds)')
plt.ylabel('Phase (radians)')
plt.legend()
plt.grid(True)
# plt.savefig("phase.png")
plt.show()

# Plot frequency spectrum before and after correction in one plot
plt.figure(figsize=(18, 6), num=("Frequency Plots"))
plt.subplot(1, 3, 1)
plt.magnitude_spectrum(signal, Fs=Fs, scale='dB')
plt.title('Frequency Spectrum Before Correction')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')   
plt.axvline(0, color='red',linewidth=0.7, linestyle='dashed')

plt.subplot(1, 3, 2)
plt.magnitude_spectrum(corrected_signal, Fs=Fs, scale='dB')
plt.title('Frequency Spectrum After Correction')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.tight_layout()
plt.axvline(0, color='red',linewidth=0.7, linestyle='dashed')

# To visualize, plot the frequency spectrums in one plot before and after correction
plt.subplot(1,3,3)
plt.magnitude_spectrum(signal, Fs=Fs, scale='dB', label='Original Signal', alpha=0.7)
plt.magnitude_spectrum(corrected_signal, Fs=Fs, scale='dB', label='Corrected Signal', alpha=0.7)
plt.title('Frequency Spectrum (Before and After Correction)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.axvline(0, color='red',linewidth=0.7, linestyle='dashed')
plt.legend()
# plt.savefig("frequency.png")
plt.show()

# Plot the points on the IQ plot for old data
plt.figure(figsize=(18, 6), num=("IQ Plots"))
plt.subplot(1, 3, 1)
plt.scatter(x_coords, y_coords, color='blue', marker='o', s=3, alpha=0.4) # Use scatter plot for points
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot Before Frequency Correction")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.axis('equal') # makes the real and imaginary axes have the same scale

# Plot the points on the IQ plot for the HW corrected frequency 
plt.subplot(1, 3, 2)
plt.scatter(corrected_x_coords, corrected_y_coords, color='orange', marker='o', s=3, alpha=0.4) # Use scatter plot for points
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot After Frequency Correction")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.axis('equal') # makes the real and imaginary axes have the same scale

# To vizualize, plot the two signals on one IQ together
plt.subplot(1,3,3)
plt.scatter(x_coords, y_coords, label='Original Signal', marker='o', s=3, alpha=0.4)
plt.scatter(corrected_x_coords, corrected_y_coords, label='Corrected Signal', marker='o', s=3, alpha=0.4)
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot Before Frequency Correction")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.axis('equal') # makes the real and imaginary axes have the same scale
plt.legend()
# plt.savefig("iq.png")
plt.show()