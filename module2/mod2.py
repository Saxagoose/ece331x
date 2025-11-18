import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#Steps
#Isolate the transmission
#Isloate I/Q information
#Plot phase vs time and magnitude vs time
#Plot I/Q data on constillation plot 



fs = 521e3 #Sampling rate 4Mhz
bufferSize = 2**16 #Buffer size
bandwidth = 26e4 #Bandwidth 
nfft = 3000
targetFreq = 0
dataSetRough = np.load("/home/goose/Documents/wpi/ece-331x/module1/data1.npy", 'r')

#Isolate Transmission
#range 8.94-9.77
dataSet = dataSetRough[int(27.97*fs):int(28.76*fs)]
#turn it into a spectogram
fPhase, tPhase, SxxPhase = signal.spectrogram(dataSet, fs=fs, window="hamming", nfft=nfft, mode="phase")
fMag, tMag, SxxMag = signal.spectrogram(dataSet, fs=fs, window="hamming", nfft=nfft, mode="magnitude")



idx = np.argmin(np.abs(fPhase - targetFreq))# Finds the index of the data freq
SxxPhase = SxxPhase[idx, :]# Isolates just that frequency, acting as a very specific bandpass
SxxMag = SxxMag[idx, :]
fig, axs = plt.subplots(2,2)
# Creates spectogram
axs[0][0].set_title("Spectrogram")
axs[0][0].specgram(dataSet, Fs=fs, NFFT=nfft)
axs[0][0].set_ylim(-4e3, 4e3)

#Creates I/Q plot
ax_polar = fig.add_subplot(2, 2, 2, polar=True)
ax_polar.set_title("I/Q Plot", va='bottom')
ax_polar.scatter(SxxPhase, SxxMag, alpha=0.7)

#Creates Magnitude and Phase Plots
xaxis = np.linspace(0, 0.757, (SxxMag.size))

axs[1][0].set_title("Signal Magnitude vs Time")
axs[1][0].set_xlabel("Time(s)")
axs[1][0].set_ylabel("Magnitude")
axs[1][0].plot(xaxis, SxxMag)

axs[1][1].plot(xaxis, SxxPhase)
axs[1][1].set_xlabel("Time(s)")
axs[1][1].set_ylabel("Phase(degrees)")
axs[1][1].set_title("Signal Phase vs Time")


plt.show()