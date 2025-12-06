from scipy import signal
import numpy as np
import bitstring as bs
import pylfsr 

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

# Additional functions for dewhitening and CRC checking
def dewhiten(packet, channel=37): 
    # Set the polynomial and initial state
    fpoly = [7, 4]
    initial_state = [int(b) for b in format(0x40 | (channel & 0x3f), '07b')]

    # Create the LFSR
    lfsr = pylfsr.LFSR(initstate=initial_state, fpoly=fpoly)
    dewhitened = bytearray()

    # Dewhiten each byte in the packet
    for byte in packet: 
        whitened_byte = 0
        # loop through each bit in byte
        for bit_pos in range(8):
            # Get the next bit from the LFSR
            whitening_bit = lfsr.state[-1]
            whitened_byte |= (whitening_bit << bit_pos)
            lfsr.next()
        
        # xor the byte with the whitenint byte
        dewhitened.append(byte ^ whitened_byte)
    
    return dewhitened

def checkCRC(packet):
    if len(packet) < 5: # Min 2 byte header and 3 byte CRC
        return False
    
    payload = packet[:-3]
    received_crc = int.from_bytes(packet[-3:], byteorder='little')

    # Calculate 24-bit CRC
    crc = 0x555555
    polynomial = 0x00065B

    # Loop through each byte in the payload
    for byte in payload:
        # xor byte into the lsb of crc
        crc ^= byte
        for _ in range(8):
            # shift right and apply polynomial if lsb is 1
            if crc & 0x000001:
                crc = (crc >> 1) ^ polynomial
            else:
                # just shift right if lsb is 0
                crc >>= 1
            
    return (((crc & 0xFFFFFF) == received_crc), received_crc)