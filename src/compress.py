"""
    Usage

    python compress.py <input features file> <feature length> <output filename>

    Example:
    python compress.py spherical-histograms.lzf 2244 data.npz

    feature length: The length of the spherical histograms. All pretrained models operate on 17*11*12=2244 dimensional features.
"""

import sys, os
import time
import numpy as np
import lzf
import struct

MAX_LEN = 4294967293 

def decompress(file_name, val_type, val_len, line_len):
    f = open(file_name, 'r')
    byte_array = f.read()
    start = time.time()
    dcomp = lzf.decompress(byte_array, MAX_LEN)
    print 'Decompress time {0}'.format(time.time() - start)
    start = time.time()
    val = struct.unpack(val_type*(len(dcomp) / val_len), dcomp)
    val = np.array(val)
    val = np.reshape(val, (len(val)/line_len, line_len))
    print 'Unpack time {0}'.format(time.time() - start)
    return val

def main(argv):
    features_path = argv[0]
    feature_len = int(argv[1])
    output_name = argv[2]

    features_path = path + features
         
    data = decompress(features_path, 'd', 8, feature_len)

    start = time.time()
    np.savez_compressed(path + '/' + output_name, data=data)
    print 'Compress time {0}'.format(time.time() - start)     
    print 'Wrote file {0}.npz in folder {1}.'.format(output_name, path)

if __name__ == '__main__':
    main(sys.argv[1:])
