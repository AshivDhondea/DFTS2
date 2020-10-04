"""
test imports
"""

import sys
sys.path.append('..')

import numpy as np
import os

from .utils import *

from models.BrokenModel import BrokenModel as BM
from .simmods import *

from .calloc import loadChannel, quantInit, plcLoader

# ---------------------------------------------------------------------------- #
import tensorflow as tf
from tensor_completion.silrtc import *


def fnRunSimul(test_str):
    print(test_str)

