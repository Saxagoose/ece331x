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
downsampleRatio = 2 #Ratio for down sampling data



# np.set_printoptions(threshold=sys.maxsize)


#load data
signal = np.load("/home/goose/Documents/wpi/ece-331x/module5/data1.npy", "r")
# signal = np.load("/Users/fionaprendergast/ECE331X/ece331x/module5/data1.npy", "r")


#Find phase difference of the data, note: downsamples 
rawPhaseDiff = p1Func.phaseDiff(signal, downsampleRatio)

#Convert signal to bit stream
bits = p1Func.convertToBits(rawPhaseDiff)

#List of advertising packets and their starting positions
packetList, adStartPos = p1Func.packetList(bits)


#Pick a packet 
startPos = adStartPos[1] #Put whatever index for whatever packet you want in here

#Dewhiten packet

#Find the end of the packet 
endPos = startPos+300*8 #put in whatever the length of the packet really is 

#CRC


#BLE frame phase data PLEASE PUT IN THE START AND END LOCATIONS WHEN WE PICK A PACKET
bleFrame = rawPhaseDiff[startPos:endPos]

#BLE frame preamble phase diff
bleFramePreamble = bleFrame[:(1*8)]

totalTime = np.linspace(0, (len(rawPhaseDiff)*downsampleRatio/fs), (len(rawPhaseDiff)))
plt.subplot(3,1,1)
plt.title("Phase Difference vs. Time for Entire Capture of Raw I/Q Data")
plt.plot(totalTime, rawPhaseDiff)
plt.xlabel("Time(s)")
plt.ylabel("Phase Difference")

frameTime = np.linspace(0, (len(bleFrame)*downsampleRatio)/fs, (len(bleFrame)))
plt.subplot(3,1,2)
plt.title("Phase Difference vs. Time of a BLE Frame")
print(np.size(frameTime))
print(np.size(bleFrame))
plt.plot(frameTime, bleFrame)
plt.xlabel("Time(s)")
plt.ylabel("Phase Difference")

preambleTime = np.linspace(0, (len(bleFramePreamble)*downsampleRatio/fs), (len(bleFramePreamble)))
plt.subplot(3,1,3)
plt.title("Phase Difference vs. Time of a BLE Frame")
plt.plot(preambleTime, bleFramePreamble)
plt.xlabel("Time(s)")
plt.ylabel("Phase Difference")


plt.show()