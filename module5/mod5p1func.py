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
    fpoly = [7, 4, 1] # x^7 + x^4 + 1

    lfsr = (channel & 0x3F) | 0b1000000 # 7 bits with MSB = 1
    # initial_state = [int(b) for b in format(0x40 | (channel & 0x3f), '07b')]
    # channel 23 (0b0101111) example = pos 0= 1, pos 1=0, pos 2=1, pos 3=1, pos 4=1, pos 5=1, pos 6=1 or 10101111
    # chanel 37 (0b100110) example = pos 0=1, pos 1=0, pos 2=0, pos 3=1, pos 4=0, pos 5=1, pos 6=1 or 1100110
    # print(f"Initial LFSR state for channel {channel}: {lfsr:07b}: compare 38 -> 1100110")

    # out = bytearray()

    # for b in packet:
    #     whiten_byte = 0
    #     for i in range(8):
    #         # Whitening bit is LSB of LFSR
    #         wbit = lfsr & 0x01
    #         whiten_byte |= (wbit << (7 - i))

    #         # Compute next LFSR state: newbit = bit6 XOR bit3
    #         newbit = ((lfsr >> 6) ^ (lfsr >> 3)) & 0x01
    #         lfsr = ((lfsr >> 1) | (newbit << 6)) & 0x7F

    #     out.append(b ^ whiten_byte)

    # return out
    # Create the LFSR
    initial_state = [(lfsr >> i) & 0x01 for i in range(7)][::-1]  # LSB first
    lfsr = pylfsr.LFSR(initstate=initial_state, fpoly=fpoly)
    dewhitened = bytearray()

    output = lfsr.runKCycle(7)  # Pre-run to set up the LFSR

    # Dewhiten each byte in the packet
    for byte in packet: 
        whitened_byte = 0
        # loop through each bit in byte
        for bit_pos in range(8):
            # Get the next bit from the LFSR
            whitening_bit = lfsr.state[-1]
            whitened_byte |= (whitening_bit << bit_pos)
        
        # xor the byte with the whitenint byte
        dewhitened.append(byte ^ whitened_byte)
    
    return dewhitened

def checkCRC(packet,received_crc):
    if len(packet) < 7: # Min 2 byte header, 2 byte pauload and 3 byte CRC
        return False
    
    payload = packet[:-3]
    # received_crc = int.from_bytes(packet[-3:], byteorder='little')

    # Calculate 24-bit CRC according to BLE spec
    # Polynomial: x^24 + x^10 + x^9 + x^6 + x^4 + x^3 + x + 1
    # Which is 0x00065B (when represented without the x^24 term)
    crc = 0x555555  # Initial value
    polynomial = 0x00065B

    # Loop through each byte in the payload
    for byte in payload:
        # xor byte into the lsb of crc
        crc ^= (byte << 16)
        for _ in range(8):
            # shift right and apply polynomial if lsb is 1
            if crc & 0x800000:
                crc = ((crc << 1) ^ polynomial) & 0xFFFFFF
            else:
                # just shift right if lsb is 0
                crc = (crc << 1) & 0xFFFFFF
    
    valid_crc = (received_crc.hex().upper() == f"{crc:06X}")
    print(f"CRC valid? {valid_crc}")
    print(f"CRC: 0x{crc:06X} vs. Received CRC: 0x{received_crc.hex().upper()}")
            
    return valid_crc, crc

def get_CRC(bits):
	# my numpy array implementation was 10x slower, I don't know why
	# this may be easier to read than the whitening, input data is handled differently
	
	exponents = [0,1,3,4,6,9,10] # from core spec, final tap is taken care of with new bit
	
	# setup registers
	state = 6*[1,0,1,0] # from core spec
	
	for bit in bits:
		new_bit = state[-1] ^ bit
		state = [0] + state[:-1]
		if new_bit == 0: continue
		for gate in exponents:
			state[gate] = state[gate]^1
		
	return state[::-1]

def check_CRC(bits, crc):
    a = get_CRC(bits)
    # Turn a into a hex number instead of list
    ahex = (a[0] << 16) | (a[1] << 8) | a[2]
    abin = ''.join(str(bit) for bit in a)
    print(f"Calculated CRC: {ahex}, {abin} \nReceived CRC: {crc.hex()}")
    return np.array_equal(a, crc)

def reverse_bits(x):
    return int('{:08b}'.format(x)[::-1], 2)

def dewhiten_ble(packet, channel=38):
    # build BLE seed
    lfsr = (channel & 0x3F) | 0x40  # BLE whitening seed
    out = bytearray()

    for b in packet:
        wb = 0
        for i in range(8):
            w = lfsr & 1
            wb |= (w << i)             # whitening applied LSB->MSB (BLE)
            newbit = ((lfsr >> 6) ^ (lfsr >> 3)) & 1
            lfsr = ((lfsr >> 1) | (newbit << 6)) & 0x7F
        out.append(b ^ wb)

    return out

# full correct wrapper for example data:
def dewhiten_wrapper(whitened_bytes, channel=38):
    # 1 – reverse bits to convert MSB text to OTA LSB order
    pkt = [reverse_bits(b) for b in whitened_bytes]

    # 2 – run BLE dewhitening
    dw = dewhiten_ble(pkt, channel)

    # 3 – reverse back for comparison with human-readable expected output
    return [reverse_bits(b) for b in dw]
