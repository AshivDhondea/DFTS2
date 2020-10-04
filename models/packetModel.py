"""
Packetization model class.

"""

"""
Created on Thu Sep 10 12:00:05 2020

Significant changes from the original DFTS.
The motivation for these changes is that we need to be able to quickly convert
backward and forward between the original data tensor and the packets.

https://stackoverflow.com/questions/6451034/python-paradigm-for-derived-fields-class-attributes-from-calculations
@author: Ashiv Hans Dhondea
"""

import numpy as np

class PacketModel(object):

    def __init__(self, **kwargs):
        """
        # Arguments
        data: 4-D tensor to be packetized
        rowsPerPacket: number of rows of the feature map to be considered as one packet
        """
        super(PacketModel, self).__init__()

        allowed_keys = {'rows_per_packet', 'data_tensor', 'packet_seq','data_shape'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)


        if 'packet_seq' not in kwargs.keys():
            # Packetize the data tensor
            self.data_shape = self.data_tensor.shape
            self.packet_seq = self.dataToPacket()
        else:
            # Unpacketize the packetized data
            if self.data_shape[1]%self.rows_per_packet ==0:
                self.numZeros = 0
            else:
                self.numZeros =  self.rows_per_packet - (self.data_shape[1]%self.rows_per_packet)

            self.data_tensor = self.packetToData()


    def dataToPacket(self):
        """ Converts 4D tensor to 5D tensor of packets

        # Arguments
            data: 4D tensor

        # Returns
            5D tensor
        """
        self.numZeros = 0

        if self.data_shape[1]%self.rows_per_packet ==0:
            self.packet_seq = np.reshape(self.data_tensor, (self.data_shape[0], -1, self.rows_per_packet, self.data_shape[2], self.data_shape[3]))
            return self.packet_seq

        self.numZeros = self.rows_per_packet - (self.data_shape[1]%self.rows_per_packet)
        zeros = np.zeros((self.data_shape[0], self.numZeros, self.data_shape[2], self.data_shape[3]))
        packets = np.concatenate((self.data_tensor, zeros), axis=1)
        self.packet_seq = np.reshape(packets, (self.data_shape[0], -1, self.rows_per_packet, self.data_shape[2], self.data_shape[3]))
        return self.packet_seq

    def packetToData(self):
        """Converts the packets back to original 4D tensor

        # Returns
            4D tensor
        """
        if self.numZeros == 0:
            self.data_tensor = np.reshape(self.packet_seq, self.data_shape)
            return self.data_tensor

        packetSeq = np.reshape(self.packet_seq, (self.data_shape[0], -1, self.data_shape[2], self.data_shape[3]))
        index  = -1*self.numZeros
        self.data_tensor = packetSeq[:, :index, :, :]
        return self.data_tensor
