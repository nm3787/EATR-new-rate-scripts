
import numpy as np
import argparse
import os,sys
sys.path.append(os.path.abspath('.'))
import rate_methods_library as RM
import json
import random
from scipy import interpolate, optimize, integrate
from scipy.stats import ks_1samp, ks_2samp
from scipy.stats import gamma as gamma_func
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='the simulation COLVAR files to check the order of', nargs='+')
parser.add_argument('-l', '--logfiles', type=str, default=None, help='the simulation PLUMED log files to check the order of', nargs='+')
parser.add_argument('-o', '--output', type=str, default='order.dat', help='the name of the output file which will contain the COLVAR files in order of Unix unpacking')

args = parser.parse_args()

if args.logfiles is not None and len(args.input) != len(args.logfiles):
    sys.exit('There are an unequal number of COLVAR files and PLUMED log files.')

with open(args.output, 'w') as f:
    if args.logfiles is None:
        for colvar in args.input:
            f.write(colvar+'\n')
    else:
        for colvar, logfile in zip(args.input,args.logfiles):
            f.write(colvar+' | '+logfile+'\n')

print("Done.")
