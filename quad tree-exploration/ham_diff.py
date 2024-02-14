#Calculate the hamming distance between two pdq length hashes.  Output is bits different. 

"""
Copyright 2023 OpenOrigins 2023

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the “Software”), to deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit 
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the 
Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import sys

def hamming_distance(hash1, hash2):
    # Convert hex strings to binary strings
    binary_hash1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
    binary_hash2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)

    # Calculate Hamming distance
    distance = sum(c1 != c2 for c1, c2 in zip(binary_hash1, binary_hash2))

    return distance

def main():
    if len(sys.argv) != 3:
        print("Usage: python hash_diff.py <hash1> <hash2>")
        sys.exit(1)

    hash1 = sys.argv[1]
    hash2 = sys.argv[2]

    if len(hash1) != len(hash2):
        print("Error: Hashes must be of the same length.")
        sys.exit(1)

    try:
        distance = hamming_distance(hash1, hash2)
        print(f"Hamming distance: {distance} bits")
    except ValueError:
        print("Error: Invalid hash format.")
        sys.exit(1)

if __name__ == "__main__":
    main()

