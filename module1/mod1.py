import numpy as np
import matplotlib.pyplot as plt
import adi 
import iio
import time


# https://analogdevicesinc.github.io/pyadi-iio/devices/adi.ad936x.html#adi.ad936x.Pluto
#Steps:
#Connect to the pluto 
#Recive data from pluto for 30 seconds
#Put data in the massive array 
#Plot with pyplot.specgram
### :)

#variables
fc = 433900000+3.8e4+1.6e3 #center freq
fs = 521e3 #Sampling rate
bufferSize = 2**16 #Buffer size
runTime = 30 #Run time
bandwidth = 26e4 #Bandwidth 
# gain = 10
# samples = runTime*fs/bufferSize #number of sample buffers
# sdr = adi.Pluto("ip:192.168.2.1")
# print("Connected to SDR!")
# #configure properties
# sdr.rx_lo = int(fc) #sets Fc
# sdr.sample_rate = int(fs) #sets sampling rate
# sdr.rx_rf_bandwidth = int(bandwidth) 
# sdr.rx_buffer_size = int(bufferSize)
# sdr.gain_control_mode_chan0 = "fast_attack"
# dataSet = np.zeros(int(samples*bufferSize), dtype=np.complex64) #creates an array of zeros of complex 64 bit numbers, to allocate space
# startTime = time.time() #Notes start time 
# for i in range(int(samples)): #collects samples until it has reached the right number of samples
#     dataSet[i*bufferSize:(i+1)*bufferSize] = sdr.rx()

# np.save("/home/goose/Documents/wpi/ece-331x/module1/data1", dataSet) #saves data
# endTime = time.time() #Notes end time so that we can know how long it ran for 
# print(f"done in {endTime-startTime}s")

#for loading data to process 
dataSet = np.load("/home/goose/Documents/wpi/ece-331x/module1/data1.npy", 'r')
#Creates spectrogram 
plt.specgram(dataSet, Fs=fs, NFFT=4096)
plt.xlabel("Time(s)")
plt.ylabel("Frequency(Hz)")
plt.show()
