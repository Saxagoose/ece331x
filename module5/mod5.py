#MOD5 MAIN LOOP
import matplotlib.pyplot as plt
import numpy as np
import mod5p1func as p1Func
import sys

# Variables 
fc = 2402e6 #center freq
fs = 4e6 #Sampling rate
bufferSize = 2**19 #Buffer size
runTime = 0.25 #Run time
bandwidth = 1e6 #Bandwidth 
channel = 37

np.set_printoptions(threshold=sys.maxsize)


#load data
signal = np.load("/home/goose/Documents/wpi/ece-331x/module5/data1.npy", "r")
# signal = np.load("/Users/fionaprendergast/ECE331X/ece331x/module5/data1.npy", "r")

#Lowpass filter
# signal = p1Func.lowpass(1e6, 101, fs, signal)

#Plot Raw I/Q of collected data 
total_time = np.linspace(0, (len(signal)/fs), (len(signal)-1))
raw_phase_diff = p1Func.phaseDiff(signal, 2)

bits = p1Func.convertToBits(raw_phase_diff)
print(p1Func.packetList(bits))
plt.plot(bits[:1000])
plt.show()