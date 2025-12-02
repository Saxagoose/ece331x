import numpy as np
import matplotlib.pyplot as plt


# Read the data from the file
signal = np.load("/Users/fionaprendergast/ECE331X/moredata/data2.npy")

# Properties
Fs = 512e3
Fc = 915000000

# Get the segment we want
print("Chopped up the signal to get what we want")
signal = signal[int(15.00*Fs):int(16.00*Fs)]
time_array = np.arange(len(signal)) / Fs

# -------------------------
# COARSE frequency correction
# -------------------------
# Square the signal to estimate frequency offset
signal_squared = signal**2 
psd = np.fft.fftshift(np.abs(np.fft.fft(signal_squared)))
f = np.linspace(-Fs/2.0, Fs/2.0, len(psd))

max_freq = f[np.argmax(psd)]
print(f"Estimated frequency offset: {max_freq/2.0} Hz")

# Apply correction to ORIGINAL signal
Ts = 1/Fs
t = np.arange(0, Ts*len(signal), Ts)
coarse_corrected_signal = signal * np.exp(-1j*2*np.pi*max_freq*t/2.0)

# -------------------------
# Costas Loop (fine correction)
# -------------------------
# N = len(coarse_corrected_signal)
# print(N)
# phase = 0
# freq = 0
# alpha = 0.132
# beta = 0.00932
# out = np.zeros(N, dtype=np.complex64)
# freq_log = []

# for i in range(N):
#     if i % 50000 == 0: 
#         print(i)
#     out[i] = coarse_corrected_signal[i] * np.exp(-1j*phase)
#     error = np.real(out[i]) * np.imag(out[i])
    
#     freq += (beta * error)
#     freq_log.append(freq * Fs / (2*np.pi))
#     phase += freq + (alpha * error)
    
#     while phase >= 2*np.pi:
#         phase -= 2*np.pi
#     while phase < 0:
#         phase += 2*np.pi

# -------------------------
# Costas Loop (fine correction) - VECTORIZED
# -------------------------
N = len(coarse_corrected_signal)
phase = np.zeros(N)
freq = np.zeros(N)
alpha = 0.332
beta = 0.0732
out = np.zeros(N, dtype=np.complex64)

print("Running Costas Loop...")
print(f"Processing {N} samples...")

for i in range(N):
    if i % 50000 == 0:  # Progress indicator
        print(f"Progress: {i}/{N} ({100*i/N:.1f}%)")
    
    out[i] = coarse_corrected_signal[i] * np.exp(-1j*phase[i])
    error = np.real(out[i]) * np.imag(out[i])
    
    if i < N-1:  # Don't go out of bounds
        freq[i+1] = freq[i] + (beta * error)
        phase[i+1] = phase[i] + freq[i+1] + (alpha * error)
        
        # Phase wrapping
        phase[i+1] = np.mod(phase[i+1], 2*np.pi)

print("Costas Loop complete!")

# Plot frequency estimate
plt.figure()
plt.plot(freq * Fs / (2*np.pi), '.-')
plt.xlabel("Sample Index")
plt.ylabel("Estimated Frequency Offset (Hz)")
plt.title("Costas Loop Frequency Estimate Over Time")
plt.grid(True)
plt.show()

# -------------------------
# IQ Plot AFTER Costas loop
# -------------------------
plt.figure(figsize=(6, 6))
plt.scatter(out.real, out.imag, marker='o', s=3, alpha=0.4)
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot After Costas Loop")
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.axis('equal')
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