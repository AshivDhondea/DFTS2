"""
Quantization model class.

Operates on the tensor which is transmitted across the channel between the device
and the cloud.

"""
import numpy as np
import time

class QLayer(object):
    """Defines the quantization layer."""
    def __init__(self, nBits):
        """
        # Arguments
            nBits: number of bits of quantization
        """
        super(QLayer, self).__init__()
        self.nBits = nBits

    def bitQuantizer(self, data):
        """Quantizes the input data to the set number of bits.

        # Arguments
            data: data to be quantized
        """
        start_time = time.time()
        self.max = np.max(data)
        self.min = np.min(data)
        np.seterr(divide='ignore', invalid='ignore')

        #refer to deep feature compression for formulae
        # self.typeSize = 'uint'+str(cpt(self.nBits))
        self.quanData = np.round(((data-self.min)/(self.max-self.min))*((2**self.nBits)-1))#.astype(self.typeSize)
        total_time = time.time() - start_time
        print(f"bit quantizer complete in {total_time}s")

    def inverseQuantizer(self):
        """Performs inverse of quantization

        # Returns
            De-Quantized data.
        """
        self.quanData = (self.quanData*(self.max-self.min)/((2**self.nBits)-1)) + self.min
        return self.quanData
