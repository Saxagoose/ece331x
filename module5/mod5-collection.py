import numpy as np
import matplotlib.pyplot as plt
import adi 
import time

#variables
fc = 2402e6 #center freq
fs = 4e6 #Sampling rate
bufferSize = 2**19 #Buffer size
runTime = 0.25 #Run time
bandwidth = 1e6 #Bandwidth 
gain = 10   
samples = runTime*fs/bufferSize #number of sample buffers
sdr = adi.Pluto("ip:192.168.2.1")
print("Connected to SDR!")
#configure properties5 000 000
sdr.rx_lo = int(fc) #sets Fc
sdr.sample_rate = int(fs) #sets sampling rate
sdr.rx_rf_bandwidth = int(fs) 
sdr.rx_buffer_size = int(bufferSize)
# sdr.gain_control_mode_chan0 = "manual"
# sdr.rx_hardwaregain_chan0 = gain
time.sleep(0.25)
dataSet = np.zeros(int(bufferSize), dtype=np.complex64) #creates an array of zeros of complex 64 bit numbers, to allocate space
dataSet = sdr.rx()

np.save("/home/goose/Documents/wpi/ece-331x/module5/data2", dataSet) #saves data

#spectrogram
plt.specgram(dataSet, Fs=fs, NFFT=4096)
plt.xlabel("Time(s)")
plt.ylabel("Frequency(Hz)")
plt.show()


# Recieve 
#   recieve at 4 MSPS
# Plot into 2-gfsk 
# Decode into binary 
# Dewhiten the data 
# Do CRC



