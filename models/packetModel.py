"""
Packetization model class.

Documentation to be sorted out later.
"""

import numpy as np
#import time # imported but unused.

class PacketModel(object):
    """Convert data to packets"""
    def __init__(self, data, rowsPerPacket):
        """
            # Arguments
                data: 4-D tensor to be packetized
                rowsPerPacket: number of rows of the feature map to be considered as one packet
        """
        super(PacketModel, self).__init__()
        self.rowsPerPacket = rowsPerPacket
        self.dataShape     = data.shape
        self.packetSeq     = self.dataToPacket(data)

    def dataToPacket(self, data):
        """ Converts 4D tensor to 5D tensor of packets

        # Arguments
            data: 4D tensor

        # Returns
            5D tensor
        """
        self.numZeros = 0

        if self.dataShape[1]%self.rowsPerPacket ==0:
            data = np.reshape(data, (self.dataShape[0], -1, self.rowsPerPacket, self.dataShape[2], self.dataShape[3]))
            return data

        self.numZeros = self.rowsPerPacket - (self.dataShape[1]%self.rowsPerPacket)
        zeros         = np.zeros((self.dataShape[0], self.numZeros, self.dataShape[2], self.dataShape[3]))
        data          = np.concatenate((data, zeros), axis=1)
        data          = np.reshape(data, (self.dataShape[0], -1, self.rowsPerPacket, self.dataShape[2], self.dataShape[3]))
        return data

    def packetToData(self):
        """Converts the packets back to original 4D tensor

        # Returns
            4D tensor
        """
        if self.numZeros == 0:
            self.packetSeq = np.reshape(self.packetSeq, self.dataShape)
            return self.packetSeq

        self.packetSeq = np.reshape(self.packetSeq, (self.dataShape[0], -1, self.dataShape[2], self.dataShape[3]))
        index          = -1*self.numZeros
        self.packetSeq = self.packetSeq[:, :index, :, :]
        return self.packetSeq
