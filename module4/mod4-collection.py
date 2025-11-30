import numpy as np
import matplotlib.pyplot as plt
import adi 
import iio
import time


#variables
fc = 915e6 #center freq
fs = 521e3 #Sampling rate
bufferSize = 2**16 #Buffer size
runTime = 30 #Run time
bandwidth = 26e4 #Bandwidth 
gain = 10   
samples = runTime*fs/bufferSize #number of sample buffers
sdr = adi.Pluto("ip:192.168.2.1")
print("Connected to SDR!")
#configure properties
sdr.rx_lo = int(fc) #sets Fc
sdr.sample_rate = int(fs) #sets sampling rate
sdr.rx_rf_bandwidth = int(fs) 
sdr.rx_buffer_size = int(bufferSize)
# sdr.gain_control_mode_chan0 = "manual"
# sdr.rx_hardwaregain_chan0 = gain
dataSet = np.zeros(int(samples*bufferSize), dtype=np.complex64) #creates an array of zeros of complex 64 bit numbers, to allocate space
startTime = time.time() #Notes start time 
for i in range(int(samples)): #collects samples until it has reached the right number of samples
    dataSet[i*bufferSize:(i+1)*bufferSize] = sdr.rx()

np.save("/home/goose/Documents/wpi/ece-331x/module4/data4", dataSet) #saves data
endTime = time.time() #Notes end time so that we can know how long it ran for 
print(f"done in {endTime-startTime}s")

#spectrogram
plt.subplot(1, 2, 1)
plt.specgram(dataSet, Fs=fs, NFFT=4096)
plt.xlabel("Time(s)")
plt.ylabel("Frequency(Hz)")



x_coords = dataSet.real
y_coords = dataSet.imag

plt.subplot(1, 2, 2)
plt.scatter(x_coords, y_coords, color='blue', marker='o', s=3, alpha=0.4) # Use scatter plot for points
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("IQ Plot")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.axis('equal') # makes the real and imaginary axes have the same scale
plt.show()

