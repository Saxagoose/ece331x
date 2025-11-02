import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
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
fc = 433900000
fs = 550e3
bufferSize = 4096*2
runTime = 30 #in seconds 
samples = runTime*fs/bufferSize #number of samples
sdr = adi.Pluto("ip:192.168.2.1")
#configure properties
sdr.rx_lo = int(fc) #sets Fc
sdr.sample_rate = int(fs) #Filtercut off
sdr.rx_rf_bandwidth = int(6e4)
sdr.rx_buffer_size = int(1024)

data = sdr.rx()
dataSet = np.array([])
startTime = time.time()
for i in range(int(samples)):
    dataSet = np.concatenate((dataSet, sdr.rx()), axis=0)
    print(i)
    time.sleep(1/(fs)*bufferSize)
    

print("done")
plt.specgram(dataSet, Fs=fs, NFFT=bufferSize)
plt.show()
