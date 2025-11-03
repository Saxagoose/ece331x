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
fs = 521e3
bufferSize = 2**16 #play with this
runTime = 30 #in seconds 
bandwidth = 26e4
samples = runTime*fs/bufferSize #number of samples
sdr = adi.Pluto("ip:192.168.2.1")
#configure properties
sdr.rx_lo = int(fc) #sets Fc
sdr.sample_rate = int(fs) #Filtercut off
sdr.rx_rf_bandwidth = int(bandwidth) #play with this
sdr.rx_buffer_size = int(bufferSize)
data = sdr.rx()
dataSet = np.zeros(int(samples*bufferSize), dtype=np.complex64)
startTime = time.time()
for i in range(int(samples)):
    dataSet[i*bufferSize:(i+1)*bufferSize] = sdr.rx()
    # time.sleep(1/(fs))

# while ((startTime-time.time())<runTime):
#     dataSet = np.concatenate((dataSet, sdr.rx()), axis=0)
#     time.sleep(1/(fs)*bufferSize)
# np.save("/home/goose/Documents/wpi/ece-331x/ece331x/module1/data")
endTime = time.time()
print(f"done in {endTime-startTime}s")
plt.specgram(dataSet, Fs=fs)
plt.xlabel("Time(s)")
plt.ylabel("Frequency(Hz)")
plt.show()
