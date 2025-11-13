import numpy as np
import matplotlib.pyplot as plt

# Load data from module 1
dataSet = np.load("/Users/fionaprendergast/ECE331X/data.npy") # Change path for your computer

# Sample rate from module 1
Fs = 512e3  # Sampling frequency from module 1

# Get the useful data from module 1 9.12-9.91 seconds
dataSet = dataSet[int(9.12*Fs):int(9.91*Fs)]

# Calculate frequency offset using FFT
N = len(dataSet)
fft_data = np.fft.fft(dataSet)
fft_freqs = np.fft.fftfreq(N, d=1/Fs)
peak_index = np.argmax(np.abs(fft_data))
frequency_offset = fft_freqs[peak_index]
print(f"Estimated Frequency Offset: {frequency_offset} Hz")

# Correct frequency offset with complex exponential function (in class 11/13)
time_array = np.arange(N) / Fs
correction_signal = np.exp(-1j * 2 * np.pi * frequency_offset * time_array)
corrected_data = dataSet * correction_signal

# Plot signal magnitude vs. time before and after correction
plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
magnitude_array = np.abs(dataSet) #Takes the magnitude of the complex numbers in the data set
time_array = np.arange(len(dataSet)) / Fs
plt.plot(time_array, magnitude_array)
plt.title('Signal Magnitude vs. Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Magnitude')
plt.grid(True)

magnitude_array = np.abs(corrected_data)
plt.subplot(1, 2, 2)
plt.plot(time_array, magnitude_array)
plt.title('Corrected Signal Magnitude vs. Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

# To visualize, plot the magnitude data on top of each other in one plot
plt.figure(figsize=(10, 6))
plt.plot(time_array, np.abs(dataSet), label='Original Signal', alpha=0.7)
plt.plot(time_array, np.abs(corrected_data), label='Corrected Signal', alpha=0.7)
plt.title('Signal Magnitude vs. Time (Before and After Correction)')
plt.xlabel('Time (seconds)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.show()

# Plot phase vs. time before and after correction
plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
phase_array = np.angle(dataSet) #Takes the phase of the complex numbers in the data set
unwrapped_phase = np.unwrap(phase_array)  # Unwrap the phase to avoid discontinuities
plt.plot(time_array, unwrapped_phase)
plt.title('Signal Phase vs. Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Phase (radians)')
plt.grid(True)  

phase_array = np.angle(corrected_data) #Takes the phase of the complex numbers in the data set
unwrapped_phase = np.unwrap(phase_array)  # Unwrap the phase to avoid discontinuities
plt.subplot(1, 2, 2)
plt.plot(time_array, unwrapped_phase)
plt.title('Corrected Signal Phase vs. Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Phase (radians)')
plt.grid(True)  
plt.show()

# To vizualize, plot the phase data on top of each other in one plot
plt.figure(figsize=(10, 6))
plt.plot(time_array, np.unwrap(np.angle(dataSet)), label='Original Signal', alpha=0.7)
plt.plot(time_array, np.unwrap(np.angle(corrected_data)), label='Corrected Signal', alpha=0.7)
plt.title('Signal Phase vs. Time (Before and After Correction)')
plt.xlabel('Time (seconds)')
plt.ylabel('Phase (radians)')
plt.legend()
plt.grid(True)
plt.show()

# Plot frequency spectrum before and after correction in one plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.magnitude_spectrum(dataSet, Fs=Fs, scale='dB')
plt.title('Frequency Spectrum Before Correction')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')   
plt.axvline(0, color='red',linewidth=0.7, linestyle='dashed')

plt.subplot(1, 2, 2)
plt.magnitude_spectrum(corrected_data, Fs=Fs, scale='dB')
plt.title('Frequency Spectrum After Correction')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.tight_layout()
plt.axvline(0, color='red',linewidth=0.7, linestyle='dashed')
plt.show()

# To visualize, plot the frequency spectrums in one plot before and after correction
plt.figure(figsize=(6, 6))
plt.magnitude_spectrum(dataSet, Fs=Fs, scale='dB', label='Original Signal', alpha=0.7)
plt.magnitude_spectrum(corrected_data, Fs=Fs, scale='dB', label='Corrected Signal', alpha=0.7)
plt.title('Frequency Spectrum (Before and After Correction)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.axvline(0, color='red',linewidth=0.7, linestyle='dashed')
plt.legend()
plt.show()

# Also plot the IQ constellation before and after correction
x_coords = dataSet.real
y_coords = dataSet.imag
corrected_x_coords = corrected_data.real
corrected_y_coords = corrected_data.imag

# Plot the points
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(x_coords, y_coords, color='blue', marker='o', s=3, alpha=0.4) # Use scatter plot for points
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot Before Frequency Correction")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.axis('equal') # makes the real and imaginary axes have the same scale

# Plot the points
plt.subplot(1, 2, 2)
plt.scatter(corrected_x_coords, corrected_y_coords, color='orange', marker='o', s=3, alpha=0.4) # Use scatter plot for points
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot After Frequency Correction")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.axis('equal') # makes the real and imaginary axes have the same scale
plt.show()