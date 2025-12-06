from scipy import signal
import numpy as np
import bitstring as bs

#Lowpass filter
def lowpass(cutoff, taps, fs, data):
    lowpass_filter = signal.firwin(taps, cutoff, fs=fs)
    filtered_data = np.convolve(lowpass_filter, data)
    return filtered_data

#Finds the phase difference over time 
def phaseDiff(data, downsample_ratio=1):
    phase = np.angle(data)
    phase_down = phase[::downsample_ratio]
    phase_unwrap = np.unwrap(phase_down)
    phase_diff = np.diff(phase_unwrap)
    return phase_diff


#Converts data to bits
def convertToBits(data):
    return np.greater(data, 0)

#Finds a bit pattern in a bit stream
# Function created by: Galahad Wernsing
# Modified for ECE 331X by: Samuel Forero
def findBitPattern(data_stream, pattern):
    pos_array = []
    data_stream = bs.Bits(data_stream)
    pattern = bs.Bits(pattern)
    positions = data_stream.findall(pattern)
    for pos in positions:
        pos_array.append(pos)
    return pos_array
	


#Finds the location of advertising packets 
def findAdPackets(data):
    preamble_and_aa = np.array([0,1,0,1,0,1,0,1,0,1,1,0,1,0,1,1,0,1,1,1,1,1,0,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1],dtype=bool)
    return findBitPattern(data, preamble_and_aa)

# Creates a list of packets in bitstrings and a list of starting position
def packetList(data):
    adPos = findAdPackets(data)
    packetList = []
    for pos in adPos:
        packet = bs.Bits(data[pos:(pos+300*8)])
        packetList.append(packet) #Most packets should fit within this size otherwise they are massive chained together 
    return  packetList, adPos
