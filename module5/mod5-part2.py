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
# The data is a list of the starting positions of each advertising packet in bits
# The packets are in the form below: 
# +-------------------+-----------------------+-----------------------+---------------+
# | Preamble (1 byte) | Access Addr (4 bytes) | Payload (2-256 bytes) | CRC (3 bytes) |
# +-------------------+-----------------------+-----------------------+---------------+

raw_phase_diff = p1Func.phaseDiff(data, 2)
bits = p1Func.convertToBits(raw_phase_diff)
packet_list, adStartPos = p1Func.packetList(bits)
print(packet_list[1].bin)
print(packet_list[1].length)

while True:
    user_input = input("Enter 'q' to quit or press Enter to continue: ")
    if user_input.lower() == 'q':
        break

dewhitened_packets = []
valid_packets = []
crcs = []

# Each position in adStartPos is the start of a packet
for packet in packet_list:
    print("\n")
    # Extract packet bits (assume max packet size of 300 bytes = 2400 bits)
    # Or use whatever max size you determined
    # end_pos = start_pos + 300 * 8  # 300 bytes * 8 bits/byte
    
    # if end_pos > len(bits):
    #     end_pos = len(bits)
    
    # packet_bits = bits[start_pos:end_pos]
    
    # Skip preamble (1 byte = 8 bits) and access address (4 bytes = 32 bits)
    # Total: 40 bits
    pdu_bits = packet[40:]

    # ================ DOUBLE CHECK THE ADDRESS ===================
    # Extract access address (4 bytes after 1-byte preamble)
    access_addr_bits = packet[8:40]  # bits 8-39
    access_addr = 0
    for i, bit in enumerate(access_addr_bits):
        access_addr |= (int(bit) << i)

    # print(f"Access Address: 0x{access_addr:08X} (should be 0x8E89BED6)")
    if access_addr != 0x8E89BED6:
        print("WARNING: Access address mismatch!")
    # print(f"Preamble and AA skipped: {packet_bits[:40]}")
    # ==============================================================
    
    # Convert bits to bytes (LSB first for BLE)
    whitened_pdu = bytearray()
    for i in range(0, len(pdu_bits), 8):
        if i + 8 <= len(pdu_bits):
            byte_bits = pdu_bits[i:i+8]
            # Convert 8 bits to byte (LSB first)
            byte_val = sum(int(bit) << bit_pos for bit_pos, bit in enumerate(byte_bits))
            whitened_pdu.append(byte_val)

    # Dewhiten data 
    channel = 37
    dewhitened_packet = p1Func.dewhiten(packet, channel)
    dewhitened_packets.append(dewhitened_packet)
    
    # # Dewhiten data 
    # channel = 37 
    # # Create LFSR and dewhiten 
	# # Based on the core specification it is 7 bit lfsr with polynomial x^7 + x^4 + 1
    # dewhitened_pdu = p1Func.dewhiten(whitened_pdu, channel)
    # dewhitened_packets.append(dewhitened_pdu)
    pdu_type = dewhitened_packet[0] & 0x0F
    length = dewhitened_packet[1]
    pdu_with_crc = dewhitened_packet[:length+3]  # PDU Type (1 byte) + Length (1 byte) + Payload + CRC
    pdu_without_crc = dewhitened_packet[:length]
    crc = dewhitened_packet[length:length+3]
    # print(f"PDU Type: 0x{pdu_type:X} (should be 0-7 for advertising)")
    print(f"Payload Length: {length} bytes")
    print(f"Rest of PDU + crc length: {len(pdu_with_crc)} bytes")
    print(f"Packet total length: {len(dewhitened_packet)} bytes")

    # TRY PROVIDED CODE INSTEAD OF MINE
    check_crc = p1Func.check_CRC(pdu_without_crc, crc)
    print(f"CRC valid? {check_crc}")
    # print(f"Calculated CRC: 0x{calc_crc:06X}")
    

    # Check CRC
    print(f"Dewhitened packet: {pdu_with_crc.hex()}")
    valid_crc, received_crc = p1Func.checkCRC(pdu_with_crc)
    valid_packets.append(valid_crc)
    crcs.append(received_crc)

    # print(f"Channel: {channel}")
    # print(f"Dewhitened packet: {pdu_with_crc.hex()}")
    # print(f"CRC valid: {valid_crc}")
    # print(f"Received CRC: 0x{received_crc:06X}")

    # if valid_crc:
    #     # Parse the packet
    #     pdu_type = pdu_with_crc[0] & 0x0F
    #     length = pdu_with_crc[1]
    #     payload = pdu_with_crc[2:2+length]
        
    #     print(f"PDU Type: 0x{pdu_type:X}")
    #     print(f"Payload length: {length}")
    #     print(f"Payload: {payload.hex()}")
	
	 



# This is all from section 3.1.1. CRC generation in the https://www.bluetooth.com/specifications/specs/core60-html/ 
# Using the polynomial x^24 + x^10 + x^9 + x^6 + x^4 + x^3 + x + 1
# Shift register is preset with 0x555555
# Check CRC of each packet




