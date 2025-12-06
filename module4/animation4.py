import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# Create output directory
output_dir = os.path.join(os.getcwd(), 'costas_output')
os.makedirs(output_dir, exist_ok=True)
print(f"Saving frames to: {output_dir}")

# Read the data from the file
signal_original = np.load("/Users/fionaprendergast/ECE331X/data0.npy")
Fs = 512e3
signal_original = signal_original[int(10.00*Fs):int(11.00*Fs)]
time_array = np.arange(len(signal_original)) / Fs

# -------------------------
# COARSE frequency correction
# -------------------------
# Square the signal to estimate frequency offset
signal_squared = signal_original**2  # Don't overwrite original!
psd = np.fft.fftshift(np.abs(np.fft.fft(signal_squared)))
f = np.linspace(-Fs/2.0, Fs/2.0, len(psd))

max_freq = f[np.argmax(psd)]
print(f"Estimated frequency offset: {max_freq/2.0} Hz")

# Apply correction to ORIGINAL signal
Ts = 1/Fs
t = np.arange(0, Ts*len(signal_original), Ts)
coarse_corrected_signal = signal_original * np.exp(-1j*2*np.pi*max_freq*t/2.0)

# -------------------------
# Costas Loop (fine correction) - VECTORIZED WITH ANIMATION
# -------------------------
N = len(coarse_corrected_signal)
phase = np.zeros(N)
freq = np.zeros(N)
alpha = 0.132
beta = 0.00932
out = np.zeros(N, dtype=np.complex64)

print("Running Costas Loop with animation...")
print(f"Processing {N} samples...")

# Animation setup
animation_interval = 50000  # Capture frame every 50k samples (adjust for more/fewer frames)
filenames = []
frame_count = 0

for i in range(N):
    if i % 100000 == 0:  # Progress indicator
        print(f"Progress: {i}/{N} ({100*i/N:.1f}%)")
    
    out[i] = coarse_corrected_signal[i] * np.exp(-1j*phase[i])
    error = np.real(out[i]) * np.imag(out[i])
    
    if i < N-1:  # Don't go out of bounds
        freq[i+1] = freq[i] + (beta * error)
        phase[i+1] = phase[i] + freq[i+1] + (alpha * error)
        
        # Phase wrapping
        phase[i+1] = np.mod(phase[i+1], 2*np.pi)
    
    # Create animation frames
    if i % animation_interval == 0 and i > 0:
        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 7))
        fig.subplots_adjust(hspace=0.4)
        
        # Plot frequency offset over time
        ax1.plot(freq[0:i] * Fs / (2*np.pi), '.-', linewidth=0.5, markersize=2)
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Freq Offset [Hz]')
        ax1.set_title('Costas Loop Frequency Estimate')
        ax1.grid(True)
        
        # Plot constellation (last 1000 points for clarity)
        plot_start = max(0, i-1000)
        ax2.scatter(np.real(out[plot_start:i]), np.imag(out[plot_start:i]), 
                   s=3, alpha=0.5)
        ax2.axis('equal')
        ax2.set_xlim([-2, 2])
        ax2.set_ylim([-2, 2])
        ax2.set_ylabel('Q (Imaginary)')
        ax2.set_xlabel('I (Real)')
        ax2.set_title(f'IQ Constellation (Sample {i}/{N})')
        ax2.grid(True)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.axvline(0, color='black', linewidth=0.5)
        
        filename = f'/tmp/costas_frame_{frame_count:04d}.png'
        frame_count += 1
        print(f"  Saving frame {frame_count}")
        fig.savefig(filename, bbox_inches='tight', dpi=100)
        filenames.append(filename)
        plt.close(fig)

print("Costas Loop complete!")
print(f"Creating animated GIF from {len(filenames)} frames...")

# Create animated gif
if len(filenames) > 0:
    images = []
    for filename in filenames:
        print(f"Loading {filename}")
        images.append(imageio.imread(filename))
    
    gif_path = os.path.join(output_dir, 'costas_correction.gif')
    imageio.mimsave(gif_path, images, fps=5)
    print(f"Animation saved to {gif_path}")
    
    # Open the directory so you can see the files
    print(f"\nFiles saved in: {output_dir}")
else:
    print("No frames captured - signal might be too short")

# -------------------------
# Final IQ Plot AFTER Costas loop
# -------------------------
plt.figure(figsize=(6, 6))
plt.scatter(out.real, out.imag, marker='o', s=3, alpha=0.4)
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot After Costas Loop (Final)")
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.axis('equal')
plt.show()