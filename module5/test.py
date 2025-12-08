#!/usr/bin/env python3

# ----------------------------------------
# Utility: reverse 8 bits in a byte
# ----------------------------------------
def reverse_bits(x):
    return int('{:08b}'.format(x)[::-1], 2)


# ----------------------------------------
# BLE Dewhitening (correct for channel=38)
# ----------------------------------------
def dewhiten_ble(packet, channel=38):
    # BLE whitening seed: bit6 = 1, bits5..0 = channel
    lfsr = (channel & 0x3F) | 0x40

    out = bytearray()

    for byte_i, b in enumerate(packet):
        whitening_byte = 0

        print(f"\nBYTE {byte_i}: input OTA={b:08b}, LFSR start={lfsr:07b}")

        for bit in range(8):
            w = lfsr & 1              # whitening bit (LSB)
            whitening_byte |= (w << bit)

            # advance LFSR (BLE spec)
            newbit = ((lfsr >> 6) ^ (lfsr >> 3)) & 1
            lfsr = ((lfsr >> 1) | (newbit << 6)) & 0x7F

            print(f"  bit {bit}: w={w}, whitening_byte={whitening_byte:08b}, LFSR={lfsr:07b}")

        unwhitened = b ^ whitening_byte
        print(f"  Final whitening byte: {whitening_byte:08b}")
        print(f"  Dewhitened OTA:       {unwhitened:08b}")

        out.append(unwhitened)

    return out


# ---------------------------------------------------------
# Test data (your example)
# ---------------------------------------------------------

POST = [
    0b00101001, 0b00110011, 0b01000111, 0b10100001, 0b10111111, 0b10111110,
    0b11000010, 0b01110010, 0b01011000, 0b11100101, 0b00110101, 0b11110111,
    0b11110011, 0b10100101
]

PRE = [
    0b01000010, 0b10010000, 0b01100101, 0b10100101, 0b00100101, 0b11000101,
    0b01000101, 0b10000011, 0b10000000, 0b01000000, 0b11000000, 0b10110101,
    0b00101101, 0b11010111
]


# ---------------------------------------------------------
# MAIN: Debugging pipeline
# ---------------------------------------------------------

print("\n=== 1. Input (Example AFTER Whitening, MSB-first) ===")
for b in POST:
    print(f"{b:08b}")

print("\n=== 2. Reverse bits (convert MSB->LSB OTA order) ===")
post_reversed = [reverse_bits(b) for b in POST]
for b in post_reversed:
    print(f"{b:08b}")

print("\n=== 3. Dewhiten (on OTA-bit-order packet) ===")
dewhitened_ota = dewhiten_ble(post_reversed, channel=38)

print("\n=== 4. Reverse bits back to human MSB format ===")
dewhitened_human = [reverse_bits(b) for b in dewhitened_ota]
for b in dewhitened_human:
    print(f"{b:08b}")

print("\n=== 5. Compare vs expected PRE-whitening bytes ===")
match = dewhitened_human == PRE
print("Match?", match)

if not match:
    print("\nDifferences:")
    for i, (a, b) in enumerate(zip(dewhitened_human, PRE)):
        print(f"{i:02d}: got={a:08b} expected={b:08b}  {'OK' if a==b else 'X'}")
