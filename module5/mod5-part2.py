import numpy as np
import matplotlib.pyplot as plt
# import time
# import sys
# import bitstring
import mod5p1func as p1Func
from pylfsr import LFSR
# from  import dewhiten

# Module 5 - Part 2: Dewhitening and CRC Checking of the collected BLE packets

# Steps to implement: 
# Load the collected data from the previous part 
# Dewhiten the data using the BLE.py functions
# Check the CRC of each packet
# Decode the valid packets 

# Get data array of bitstring objects (each is one packet (267 bytes) but we kept 300 bytes to be safe)
data = np.load("/Users/fionaprendergast/ECE331X/ece331x/module5/data1.npy", "r")
# In the form of this below: 
# +-------------------+-----------------------+-----------------------+---------------+
# | Preamble (1 byte) | Access Addr (4 bytes) | Payload (2-256 bytes) | CRC (3 bytes) |
# +-------------------+-----------------------+-----------------------+---------------+

raw_phase_diff = p1Func.phaseDiff(data, 2)
bits = p1Func.convertToBits(raw_phase_diff)
packet_list = p1Func.packetList(bits)
# print(packet_list)

dewhitened_packets = []
valid_packets = []
crcs = []

# Cut out the preambles and access addresss (1 byte, 4 bytes) 
for whitened_packet in packet_list: 
    whitened_pdu = whitened_packet[5:] # Now each packet starts at the payload
    
    # Dewhiten data 
    channel = 37 
    # Create LFSR and dewhiten 
	# Based on the core specification it is 7 bit lfsr with polynomial x^7 + x^4 + 1
    dewhitened_pdu = p1Func.dewhiten(whitened_pdu, channel)
    dewhitened_packets.append(dewhitened_pdu)

    # get the length of the payload from the second byte
    length = dewhitened_pdu[1]
    pdu_with_crc = dewhitened_pdu[2:length + 3]  # PDU Type (1 byte) + Length (1 byte) + Payload + CRC

    # Check CRC
    valid_crc, received_crc = p1Func.checkCRC(pdu_with_crc)
    valid_packets.append(valid_crc)
    crcs.append(received_crc)

    # print(f"Channel: {channel}")
    print(f"Dewhitened packet: {pdu_with_crc.hex()}")
    print(f"CRC valid: {valid_crc}")
    print(f"Received CRC: 0x{received_crc:06X}")

    if valid_crc:
        # Parse the packet
        pdu_type = pdu_with_crc[0] & 0x0F
        length = pdu_with_crc[1]
        payload = pdu_with_crc[2:2+length]
        
        print(f"PDU Type: 0x{pdu_type:X}")
        print(f"Payload length: {length}")
        print(f"Payload: {payload.hex()}")
	
	 



# This is all from section 3.1.1. CRC generation in the https://www.bluetooth.com/specifications/specs/core60-html/ 
# Using the polynomial x^24 + x^10 + x^9 + x^6 + x^4 + x^3 + x + 1
# Shift register is preset with 0x555555
# Check CRC of each packet




