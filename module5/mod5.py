#MOD5 MAIN LOOP
import matplotlib.pyplot as plt
import numpy as np
import mod5p1func as p1Func
import sys

# Variables 
fc = 2404e6 #center freq
fs = 20e6 #Sampling rate
bufferSize = 2**23 #Buffer size
runTime = 0.25 #Run time
bandwidth = 1e6 #Bandwidth 

np.set_printoptions(threshold=sys.maxsize)

#load data
signal = np.load("/home/goose/Documents/wpi/ece-331x/module5/data1.npy", "r")

#Lowpass filter
# signal = p1Func.lowpass(1e6, 101, fs, signal)

#Plot Raw I/Q of collected data 
total_time = np.linspace(0, (len(signal)/fs), (len(signal)-1))
raw_phase_diff = p1Func.phaseDiff(signal, 2)

bits = p1Func.convertToBits(raw_phase_diff)
bits = np.append(bits, [0,1,0,1,0,1,0,1,0,1,1,0,1,0,1,1,0,1,1,1,1,1,0,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1])
print(p1Func.findAdPackets(bits))
plt.plot(bits[:1000])
plt.show()