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
#:)

#variables
fc = 433900000
fs = fc*2
runTime = 30 #in seconds 
samples = 30*fs #number of samples
sdr = adi.Pluto("ip:192.168.2.1")
#configure properties
rx_lo = int(fc) #sets Fc
sample_rate = int(fs) #Filtercut off
rx_rf_bandwidth = int(fs)
rx_buffer_size = int(1024)

data = sdr.rx()
dataSet = np.array([])
startTime = time.time()
for i in range(fs):
    dataSet = np.concatenate((dataSet, sdr.rx()), axis=0)
    time.sleep(1/fs)

print("done")
plt.specgram(dataSet, Fs=fs)
plt.show()
