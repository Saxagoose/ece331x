#MOD5 MAIN LOOP
import matplotlib.pyplot as plt
import numpy as np
import mod5p1func as p1Func

# Variables 
fc = 2404e6 #center freq
fs = 20e6 #Sampling rate
bufferSize = 2**23 #Buffer size
runTime = 0.25 #Run time
bandwidth = 1e6 #Bandwidth 



#load data
signal = np.load("/home/goose/Documents/wpi/ece-331x/module5/data0.npy", "r")
plt.specgram()

#Lowpass filter
signal = p1Func.lowpass(1e6, 101, fs, signal)

#Plot Raw I/Q of collected data 
total_time = np.linspace(0, (len(signal)/fs), (len(signal)-1))
raw_phase_diff = p1Func.phaseDiff(signal)

plt.plot(raw_phase_diff)
plt.show()

