import sys
import threading
import cffi
import numpy as np
import asdf
from tic import tic

def generate_dB(infile):
    h = infile.tree['h']
    scales = infile.tree['scales']
    seeds = infile.tree['seeds']
    
