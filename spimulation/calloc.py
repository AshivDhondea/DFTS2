"""
Call allocator.

Documentation to be sorted out later.

"""

import sys
sys.path.append('..')

from channel.gbChannel import GBC
from channel.trivialChannel import RLC
from models.quantizer import QLayer as QL
from plc import linearInterp, nearestNeighbours

def quantInit(quantization):
    """Selects quantization based on the user's choice

    # Arguments
        quantization: dictionary containing user's options

    # Returns
        QLayer object or string if quantization is turned off
    """
    if quantization['include']:
        return QL(quantization['numberOfBits'])
    else:
        return 'noQuant'

def loadChannel(channel):
    """Selects channel based on the user's choice

    # Arguments
        channel: dictionary containing user's options

    # Returns
        channel object or string if channel option is turned off
    """
    chtype = list(channel.keys())[0]

    if chtype==0:
        return 'noChannel'
    elif chtype=='GilbertChannel':
        lp = channel[chtype]['lossProbability']
        bl = channel[chtype]['burstLength']
        return GBC(lp, bl)
    elif chtype=='RandomLossChannel':
        lp = channel[chtype]['lossProbability']
        return RLC(lp)

def plcLoader(lossConceal):
    """Selects loss concealment based on the user's choice

    # Arguments
        lossConceal: dictionary containing user's options

    # Returns
        Function object or string if loss concealment is turned off
    """
    chtype = list(lossConceal.keys())[0]

    if chtype == 0:
        return 'noConceal'
    elif chtype== 'Linear':
        return linearInterp.interpPackets
    elif chtype== 'nearestNeighbours':
        return nearestNeighbours.interpPackets
