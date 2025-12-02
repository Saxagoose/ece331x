import numpy as np
import matplotlib.pyplot as plt

# Read the data from the file that we saved it to 
signal = np.load("/Users/fionaprendergast/ECE331X/data0.npy") # Change path for your computer

# Properties of the data we collected 
Fs = 512e3 # Sample rate
Fc = 915000000 # Center Frequency 915 MHz

# Get the part of the signal that has data that we want
print("Chopped up the signal to get what we want")
signal = signal[int(10.00*Fs):int(11.00*Fs)]
time_array = np.arange(len(signal)) / Fs

mag = np.abs(signal)
# plt.figure(figsize=(10,6))
plt.plot(time_array, mag, label='Original Signal', alpha=0.7)
plt.title('Signal Magnitude vs. Time ')
plt.xlabel('Time (seconds)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
# plt.savefig("both_magnitude.png")
plt.show()

#Creates spectrogram (module 1)
plt.specgram(signal, Fs=Fs)
plt.xlabel("Time(s)")
plt.ylabel("Frequency(Hz)")
plt.show()

# -------------------------
# IQ Plot BEFORE correction
# -------------------------
x_coords = signal.real
y_coords = signal.imag

# Plot the points on the IQ plot for old data
plt.figure(figsize=(6, 6), num=("IQ Plot - Before Coarse Correction"))
plt.scatter(x_coords, y_coords, color='blue', marker='o', s=3, alpha=0.4) # Use scatter plot for points
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot Before Frequency Correction")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.axis('equal') # makes the real and imaginary axes have the same scale
plt.show()

# -------------------------
# COARSE frequency correction
# -------------------------
psd = np.fft.fftshift(np.abs(np.fft.fft(signal)))
f = np.linspace(-Fs/2.0, Fs/2.0, len(psd))

plt.figure(figsize=(12,6), num="fxmag")
plt.subplot(1,2,2)
plt.plot(f, psd)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("PSD Before Coarse Correction")
plt.grid(True)

max_freq = f[np.argmax(psd)]
Ts = 1/Fs # calc sample period
t = np.arange(0, Ts*len(signal), Ts) # create time vector
coarse_corrected_signal = signal * np.exp(-1j*2*np.pi*max_freq*t)

psd_corr = np.fft.fftshift(np.abs(np.fft.fft(coarse_corrected_signal)))

plt.subplot(2,2,2)
plt.plot(f, psd_corr)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("PSD After Coarse Correction")
plt.grid(True)
plt.show()

# Plot IQ after COARSE Correction
corrected_x_coords = coarse_corrected_signal.real
corrected_y_coords = coarse_corrected_signal.imag

plt.figure(figsize=(6, 6), num=("IQ Plot - After Coarse Correction"))
plt.scatter(corrected_x_coords, corrected_y_coords, color='blue', marker='o', s=3, alpha=0.4) # Use scatter plot for points
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot After Coarse Correction")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.axis('equal') # makes the real and imaginary axes have the same scale
plt.show()

# -------------------------
# Costas Loop (fine correction)
# -------------------------
# This is the Costas Loop Code from Chapter 17 https://pysdr.org/content/sync.html
N = len(coarse_corrected_signal)
phase = 0
freq = 0
# These next two params is what to adjust, to make the feedback loop faster or slower (which impacts stability)
alpha = 0.132
beta = 0.00932
out = np.zeros(N, dtype=np.complex64)
freq_log = []
for i in range(N):
    out[i] = coarse_corrected_signal[i] * np.exp(-1j*phase) # adjust the input sample by the inverse of the estimated phase offset
    error = np.real(out[i]) * np.imag(out[i]) # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)

    # Advance the loop (recalc phase and freq offset)
    freq += (beta * error)
    freq_log.append(freq * Fs / (2*np.pi)) # convert from angular velocity to Hz for logging
    phase += freq + (alpha * error)

    # Optional: Adjust phase so its always between 0 and 2pi, recall that phase wraps around every 2pi
    while phase >= 2*np.pi:
        phase -= 2*np.pi
    while phase < 0:
        phase += 2*np.pi

# -------------------------
# Plot Costas loop frequency estimate
# -------------------------
plt.figure()
plt.plot(freq_log, '.-')
plt.xlabel("Sample Index")
plt.ylabel("Estimated Frequency Offset (Hz)")
plt.title("Costas Loop Frequency Estimate Over Time")
plt.grid(True)
plt.show()

# -------------------------
# IQ Plot AFTER Costas loop
# -------------------------
fine_corrected_x = out.real
fine_corrected_y = out.imag

plt.figure(figsize=(6, 6), num="IQ Plot - After Costas Loop")
plt.scatter(fine_corrected_x, fine_corrected_y, marker='o', s=3, alpha=0.4)
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot After Costas Loop (Fine Frequency/Phase Correction)")
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.axis('equal')
plt.show()