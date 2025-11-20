import numpy as np
import matplotlib.pyplot as plt

# Load data from module 1
dataSet = np.load("/Users/fionaprendergast/ECE331X/data.npy") # Change path for your computer

# Sample rate from module 1
Fs = 512e3  # Sampling frequency from module 1

# Get the useful data from module 1 9.12-9.91 seconds
dataSet = dataSet[int(9.12*Fs):int(9.91*Fs)]

#Creates spectrogram (module 1)
# plt.specgram(dataSet, Fs=Fs)
# plt.xlabel("Time(s)")
# plt.ylabel("Frequency(Hz)")
# plt.show()

# MODULE 2 CODE
magnitude_array = np.abs(dataSet) #Takes the magnitude of the complex numbers in the data set
time_array = np.arange(len(dataSet)) / Fs
plt.figure(figsize=(10, 6))
plt.plot(time_array, magnitude_array)
plt.title('Signal Magnitude vs. Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

phase_array = np.angle(dataSet) #Takes the phase of the complex numbers in the data set
unwrapped_phase = np.unwrap(phase_array)  # Unwrap the phase to avoid discontinuities
plt.figure(figsize=(10, 6))
plt.plot(time_array, unwrapped_phase)
plt.title('Signal Phase vs. Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Phase (radians)')
plt.grid(True)
plt.show()

# Make the IQ plot 
# Get the coordinates
x_coords = dataSet.real
y_coords = dataSet.imag

# Plot the points
plt.figure(figsize=(6, 6))
plt.scatter(x_coords, y_coords, color='blue', marker='o', s=3, alpha=0.4) # Use scatter plot for points
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.axis('equal') # makes the real and imaginary axes have the same scale
plt.show()

# Standardize IQ
# Standardize (zero-mean, unit-variance)
x_std = (x_coords - np.mean(x_coords)) / np.std(x_coords)
y_std = (y_coords - np.mean(y_coords)) / np.std(y_coords)

plt.figure(figsize=(6, 6))
plt.scatter(x_std, y_std, s=3, alpha=0.4)
plt.xlabel("Real Axis (standardized)")
plt.ylabel("Imaginary Axis (standardized)")
plt.title("Standardized IQ Plot")
plt.grid(True)
plt.axhline(0, linewidth=0.5, color='black')
plt.axvline(0, linewidth=0.5, color='black')
plt.axis('equal')
plt.show()