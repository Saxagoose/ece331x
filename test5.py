import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 

# -------------------------
# Load data and basic setup
# -------------------------
signal = np.load("/Users/fionaprendergast/ECE331X/filtered_data.npy")
signal_flipped = signal * np.exp(-1j * np.pi)

# Fs = 512e3
# Fc = 915_000_000
# numtaps = 251
# cuttoff = 32000
# cuttoff_bp = [1.15e4, 1.73e4]
# graph_range = 700
# iq_graph_axis = [-graph_range, graph_range, -graph_range, graph_range]

# # Apply a bandpass filter to the signal
# bandpass_filter = signal.firwin(numtaps, cuttoff_bp, fs=Fs, pass_zero=False)
# filtered_signal = np.convolve(data, bandpass_filter)

# # Choose the segment of interest (adjust if needed after looking at |signal|)
# print("Chopped up the signal to get what we want")
# filtered_signal = filtered_signal[int(15.00 * Fs):int(20.00 * Fs)]
# time_array = np.arange(len(filtered_signal)) / Fs

# # -------------------------
# # IQ Plot BEFORE correction
# # -------------------------
# x_coords = filtered_signal.real
# y_coords = filtered_signal.imag

# # Plot the points on the IQ plot for old data
# plt.figure(figsize=(6, 6), num=("IQ Plot - Before Coarse Correction"))
# plt.scatter(x_coords, y_coords, color='blue', marker='o', s=3, alpha=0.4) # Use scatter plot for points
# plt.xlabel("Real Axis")
# plt.ylabel("Imaginary Axis")
# plt.title("IQ Plot Before Frequency Correction")
# plt.grid(True)
# plt.axhline(0, color='black',linewidth=0.5)
# plt.axvline(0, color='black',linewidth=0.5)
# plt.axis('equal') # makes the real and imaginary axes have the same scale
# plt.show()

# # -------------------------
# # COARSE frequency correction (using squaring method for BPSK)
# # -------------------------
# signal_squared = filtered_signal ** 2
# psd = np.fft.fftshift(np.abs(np.fft.fft(signal_squared)))
# f = np.linspace(-Fs / 2.0, Fs / 2.0, len(psd), endpoint=False)

# plt.figure(figsize=(6,6), num="PSD Before and After Coarse Correction")

# # Top: PSD before coarse correction
# plt.subplot(2, 1, 1)
# plt.plot(f, psd)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude")
# # plt.title("PSD Before Coarse Correction")
# plt.grid(True)

# max_freq = f[np.argmax(psd)]
# coarse_offset = max_freq / 2.0
# print(f"Estimated frequency offset (coarse, Hz): {coarse_offset}")

# Ts = 1.0 / Fs
# t = np.arange(len(filtered_signal)) * Ts

# # Correct original (unsquared) signal
# coarse_corrected_signal = filtered_signal * np.exp(-1j * 2 * np.pi * coarse_offset * t / 2.0)

# # Bottom: PSD after coarse correction
# psd_corr = np.fft.fftshift(np.abs(np.fft.fft(coarse_corrected_signal)))
# # check_signal = coarse_corrected_signal ** 2
# # psd_check = np.fft.fftshift(np.abs(np.fft.fft(check_signal)))
# plt.subplot(2, 1, 2)
# plt.plot(f, psd_corr)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude")
# # plt.title("PSD After Coarse Correction")
# plt.grid(True)

# plt.tight_layout()
# plt.show()


# # -------------------------
# # DPLL / Costas Loop definitions
# # -------------------------

# def mixer(dds_out, sample):
#     """Multiply DDS output (complex oscillator) with input sample."""
#     return sample * dds_out

# def get_phase_error_bpsk(sample):
#     """
#     Phase error detector for BPSK Costas loop.
#     Standard: error = Re(y) * Im(y).
#     """
#     return np.real(sample) * np.imag(sample)

# def lpf_error(prev_filtered, new_error, gamma):
#     """
#     First-order IIR low-pass filter on the phase error.
#     filtered[n] = filtered[n-1] + gamma * (new_error - filtered[n-1])
#     """
#     return prev_filtered + gamma * (new_error - prev_filtered)

# def dds_step(phase, freq):
#     """
#     DDS step: advance phase by freq (radians/sample) and return oscillator.
#     phase_out = phase + freq
#     DDS output = exp(-j * phase_out)
#     """
#     phase_out = phase + freq
#     phase_out = np.mod(phase_out, 2 * np.pi)  # keep in [0, 2π)
#     dds_out = np.exp(-1j * phase_out)
#     return phase_out, dds_out

# # -------------------------
# # Costas Loop (fine correction) with explicit DPLL structure
# # -------------------------
# loop_in = coarse_corrected_signal
# N = len(loop_in)

# # Loop state
# phase = np.zeros(N, dtype=np.float64)   # DDS phase
# freq = np.zeros(N, dtype=np.float64)    # DDS frequency (radians/sample)
# out = np.zeros(N, dtype=np.complex64)   # mixed (corrected) output
# error_raw = np.zeros(N, dtype=np.float64)
# error_filt = np.zeros(N, dtype=np.float64)

# # Loop coefficients (tune as needed)
# # Start more conservative than the original values for stability
# alpha = 0.98      # proportional term (phase)
# beta = 0.065     # integral term (frequency)
# gamma = 0.022      # error LPF smoothing

# print("Running Costas Loop...")
# print(f"Processing {N} samples...")

# # Initialize DDS phase, freq, and filtered error
# phase_curr = 0.0
# freq_curr = 0.0
# err_filt_curr = 0.0

# for i in range(N):
#     if i % 50_000 == 0:
#         print(f"Progress: {i}/{N} ({100.0 * i / N:.1f}%)")

#     # DDS: generate oscillator and update phase
#     phase_curr, dds_out = dds_step(phase_curr, freq_curr)

#     # Mixer: rotate input by DDS
#     mixed = mixer(dds_out, loop_in[i])
#     out[i] = mixed

#     # Phase error detector for BPSK
#     err = get_phase_error_bpsk(mixed)
#     error_raw[i] = err

#     # Low-pass filter on error
#     err_filt_curr = lpf_error(err_filt_curr, err, gamma)
#     error_filt[i] = err_filt_curr

#     # Loop filter / update_dds:
#     # PI controller: freq += beta * err_filt; phase is already advanced by freq,
#     # but we also apply proportional correction through alpha * err_filt
#     freq_curr += beta * err_filt_curr
#     # Incorporate proportional correction directly into phase state
#     phase_curr = np.mod(phase_curr + alpha * err_filt_curr, 2 * np.pi)

#     # Log state
#     freq[i] = freq_curr
#     phase[i] = phase_curr

# print("Costas Loop complete!")

# # -------------------------
# # Costas Loop (fine correction) - VECTORIZED
# # -------------------------
# # N = len(coarse_corrected_signal)
# # phase = np.zeros(N)
# # freq = np.zeros(N)
# # alpha = 0.332
# # beta = 0.0732
# # out = np.zeros(N, dtype=np.complex64)

# # print("Running Costas Loop...")
# # print(f"Processing {N} samples...")

# # for i in range(N):
# #     if i % 50000 == 0:  # Progress indicator
# #         print(f"Progress: {i}/{N} ({100*i/N:.1f}%)")
    
# #     out[i] = coarse_corrected_signal[i] * np.exp(-1j*phase[i])
# #     error = np.real(out[i]) * np.imag(out[i])
    
# #     if i < N-1:  # Don't go out of bounds
# #         freq[i+1] = freq[i] + (beta * error)
# #         phase[i+1] = phase[i] + freq[i+1] + (alpha * error)
        
# #         # Phase wrapping
# #         phase[i+1] = np.mod(phase[i+1], 2*np.pi)

# # print("Costas Loop complete!")

# # -------------------------
# # Plot frequency estimate (in Hz)
# # -------------------------
# freq_hz = freq * Fs / (2 * np.pi)

# plt.figure()
# plt.plot(freq_hz, '.-', markersize=2)
# plt.xlabel("Sample Index")
# plt.ylabel("Estimated Frequency Offset (Hz)")
# plt.title("Costas Loop Frequency Estimate Over Time")
# plt.grid(True)
# plt.show()

# # -------------------------
# # IQ Plot AFTER Costas loop
# # -------------------------
# plt.figure(figsize=(6, 6))
# plt.scatter(out.real, out.imag, marker='o', s=3, alpha=0.4)
# plt.xlabel("Real Axis")
# plt.ylabel("Imaginary Axis")
# plt.title("IQ Plot After Costas Loop (DPLL)")
# plt.grid(True)
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
# plt.axis('equal')
# plt.show()

# plt.scatter(out.real, out.imag, color='blue', marker='o', s=3, alpha=0.01) # Use scatter plot for points
# # plt.hexbin(np.real(out), np.imag(out), gridsize=1000)
# plt.xlabel("Real Axis")
# plt.ylabel("Imaginary Axis")
# plt.title("IQ Plot of Fine and Course Frequency Corrected and Filtered Samples")
# plt.grid(True)
# plt.axhline(0, color='black',linewidth=0.5)
# plt.axvline(0, color='black',linewidth=0.5)
# plt.axis(iq_graph_axis) # makes the real and imaginary axes have the same scale 
# plt.show()

# -------------------------
# Symbol timing recovery and decoding (simple, for testing)
# -------------------------
samples_per_symbol = 128  # adjust if needed

# Skip initial quarter so loop can settle
N = len(signal_flipped)
start_sample = N // 4
symbols = signal_flipped[start_sample::samples_per_symbol]

# for sps in range(1, 50):
#     nbits = len(signal[N//4::sps])
#     print(sps, nbits, nbits % 8)
working_vals = []

for sps in [1, 2, 4, 8, 16, 32, 64, 128]:
    best = None
    print(f"=======", sps, "samples per symbol ========")
    for offset in range(sps):
        symbols = signal_flipped[N//4 + offset :: sps]
        bits = (np.real(symbols) > 0).astype(int)
        print(offset, len(bits), len(bits) % 8)
        if len(bits) % 8: 
            working_vals.append((sps,offset))

    # bits = (np.real(signal[N//4::sps]) > 0).astype(int)
    # print(sps, len(bits), len(bits) % 8)


# BPSK decision: sign of the real part
bits = (np.real(symbols) > 0).astype(int)

print(f"Number of decoded bits: {len(bits)}")

if len(bits) % 8 == 0 and len(bits) > 0:
    bytes_array = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        byte_val = int(''.join(map(str, byte)), 2)
        bytes_array.append(byte_val)

    message_chars = [chr(b) for b in bytes_array if 32 <= b <= 126]
    message = ''.join(message_chars)
    print(f"Decoded message (printable ASCII only): {message}")
else:
    print("Bit length not a multiple of 8 or no bits; skipping ASCII decode.")
