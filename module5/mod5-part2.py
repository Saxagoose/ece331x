import numpy as np
import matplotlib.pyplot as plt
# import time
# import sys
# import bitstring
import mod5p1func as p1Func
from pylfsr import LFSR
import os
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

# ============= Sanity check dewhitening function ============
# channel = 38
# test = os.urandom(16)
# w = p1Func.dewhiten(test, channel)      # whitening (same as dewhitening)
# d = p1Func.dewhiten(w, channel)         # dewhiten

# print("Self-test OK?", d == test)
# print("original: ", [bin(b) for b in test])
# print("whitened: ", [bin(b) for b in w])
# print("back:     ", [bin(b) for b in d])
# =============================================================

# Check to see if my code sucks or nah
# example packet from section 6.4.2 of the Core Spec 6.0
# PDU and CRC before whitening:
# 01000010 10010000 01100101 10100101 00100101 11000101 01000101
# 10000011 10000000 01000000 11000000 10110101 00101101 11010111
example_bits = [0,1,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,1,1,0,0,1,0,1,1,0,1,0,0,1,0,1,0,0,1,0,0,1,0,1,1,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,
1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,1,1,0,1,0,1,1,1]
example_not_white_bytes = [0b01000010, 0b10010000, 0b01100101, 0b10100101, 0b00100101, 0b11000101, 0b01000101,
                          0b10000011, 0b10000000, 0b01000000, 0b11000000, 0b10110101, 0b00101101, 0b11010111]
# example_bytearray = bytearray([0o01000010, 0o10010000, 0o01100101, 0o10100101, 0o00100101,
#                                0o11000101, 0o01000101, 0o10000011, 0o10000000,
#                                0o01000000, 0o11000000, 0o10110101, 0o00101101,
#                                0o11010111])

# PDU and CRC after whitening:
whitened_bytes = [0b00101001, 0b00110011, 0b01000111, 0b10100001, 0b10111111, 0b10111110, 0b11000010,
                 0b01110010, 0b01011000, 0b11100101, 0b00110101, 0b11110111, 0b11110011, 0b10100101]
whitened_bits = [0,0,1,0,1,0,0,1,0,0,1,1,0,0,1,1,0,1,0,0,0,1,1,1,1,0,1,0,0,0,0,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,0,0,0,0,1,0,
                 0,1,1,1,0,0,1,0,0,1,0,1,1,0,0,0,1,1,1,0,0,1,0,1,0,0,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,1,1,1,0,1,0,0,1,0,1]
# Convert bits to bytes (LSB first for BLE)
# whitened_example_pdu = bytearray()
# for i in range(0, len(whitened_bits), 8):
#     if i + 8 <= len(whitened_bits):
#         byte_bits = whitened_bits[i:i+8]
#         # Convert 8 bits to byte (LSB first)
#         byte_val = sum(int(bit) << bit_pos for bit_pos, bit in enumerate(byte_bits))
#         whitened_example_pdu.append(byte_val)

result = p1Func.dewhiten_wrapper(whitened_bytes, 38)
print("Dewhitening function works correctly?", result == example_not_white_bytes)
print(result)
print(example_not_white_bytes)


packet_corrected = [p1Func.reverse_bits(b) for b in whitened_bytes]
example_bytes = p1Func.dewhiten(packet_corrected, channel=38)
dw_msb = [p1Func.reverse_bits(b) for b in example_bytes]
# print(f"Whitened example: {[f'{byte:08b}' for byte in whitened_example_pdu]}")
print(f"Dewhitened example matches expected? {dw_msb == example_not_white_bytes}")
print(f"Calsulated dewhitened from example: {[f'{byte:08b}' for byte in dw_msb]}")
print(f"Expected dewhitened: {[f'{byte:08b}' for byte in example_not_white_bytes]}\n")

polynomial = 0x00065B
# test = example_bytes % polynomial
# print(f"Test mod polynomial: {test}")

check_example = p1Func.checkCRC(example_bytes, bytearray([(0b10110101), (0b00101101), (0b11010111)]))
print(f"Example packet CRC valid? {check_example}")

while True:
    user_input = input("Enter 'q' to quit or press Enter to continue: ")
    if user_input.lower() == 'q':
        break

dewhitened_packets = []
valid_packets = []
crcs = []

# Each position in adStartPos is the start of a packet
for start_pos, packet in zip(adStartPos, packet_list):
# for packet in packet_list:
    print("\n")
    # Extract packet bits (assume max packet size of 300 bytes = 2400 bits)
    # Or use whatever max size you determined
    end_pos = start_pos + 300 * 8  # 300 bytes * 8 bits/byte
    
    if end_pos > len(bits):
        end_pos = len(bits)
    
    packet_bits = bits[start_pos:end_pos]
    
    # Skip preamble (1 byte = 8 bits) and access address (4 bytes = 32 bits)
    # Total: 40 bits
    pdu_bits = packet_bits[40:] # use the start pos thing
    # pdu_bits = packet[40:] # use the packet directly

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
    
    # Get the packet type and length
    pdu_type = dewhitened_packet[0] & 0x0F
    length = dewhitened_packet[1]
    # print(f"PDU Type: 0x{pdu_type:X} (should be 0-7 for advertising)")

    # Get the payload data and CRC
    pdu_with_crc = dewhitened_packet[:length+3]  # PDU Type (1 byte) + Length (1 byte) + Payload + CRC
    pdu_without_crc = dewhitened_packet[:length]
    crc = dewhitened_packet[length:length+3]

    # Check with printouts
    # print(f"PDU Type: 0x{pdu_type:X} (should be 0-7 for advertising)")
    print(f"Payload Length: {length} bytes, with CRC: {len(pdu_with_crc)} bytes")
    # print(f"Packet total length: {len(dewhitened_packet)} bytes")

    # Check CRC
    print(f"Dewhitened packet: {pdu_with_crc.hex()}")
    valid_crc, received_crc = p1Func.checkCRC(pdu_with_crc, crc)
    valid_packets.append(valid_crc)
    crcs.append(received_crc)

    # TRY PROVIDED CODE INSTEAD OF MINE
    # check_crc = p1Func.check_CRC(pdu_without_crc, crc)
    # print(f"CRC valid? {check_crc}")
    # # print(f"Calculated CRC: 0x{calc_crc:06X}")

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




