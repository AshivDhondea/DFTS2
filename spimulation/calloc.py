"""
Call allocator.

Documentation to be sorted out later.

"""

import sys
sys.path.append('..')

from channel.gbChannel import GBC
from channel.trivialChannel import RLC
from models.quantizer import QLayer as QL
# ---------------------------------------------------------------------------- #
def quantInit(quantization,tensor_id='default_for_testConfig'):
    """Select quantization based on the user's choice.

    # Arguments
        quantization: dictionary containing user's options
        tensor_id: integer representing index of tensor to be packetized.

    # Returns
        QLayer object or string if quantization is turned off
    """
    if quantization['include']:
        if tensor_id == 'default_for_testConfig':
            return QL(quantization['numberOfBits'])
        else:
            return QL(quantization[tensor_id]['numberOfBits'])
    else:
        return 'noQuant'

def loadChannel(channel):
    """Select channel based on the user's choice.

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
