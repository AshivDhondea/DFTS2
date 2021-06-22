"""
Quantization model class.

Operates on the tensor which is transmitted across the channel between the device
and the cloud.

Changelog:
27 December 2020: Use timeit instead of time to measure execution time.

"""
import numpy as np
from timeit import default_timer as timer

class QLayer(object):
    """Define the quantization layer."""
    
    def __init__(self, nBits):
        """
        Initialize a quantization object class.
        
        # Arguments
            nBits: number of bits of quantization
        """
        super(QLayer, self).__init__()
        self.nBits = nBits

    def bitQuantizer(self, data):
        """
        Quantize the input data to the set number of bits.

        # Arguments
            data: data to be quantized
        """
        # Use timeit instead of time.
        start_time = timer()
        self.max = np.max(data)
        self.min = np.min(data)
        np.seterr(divide='ignore', invalid='ignore')

        #refer to deep feature compression for formulae
        # self.typeSize = 'uint'+str(cpt(self.nBits))
        self.quanData = np.round(((data-self.min)/(self.max-self.min))*((2**self.nBits)-1))#.astype(self.typeSize)
        total_time = timer() - start_time
        print(f"Bit quantization complete in {total_time:.3f}s")

    def inverseQuantizer(self):
        """
        Perform inverse of quantization.

        # Returns
            De-Quantized data.
        """
        self.quanData = (self.quanData*(self.max-self.min)/((2**self.nBits)-1)) + self.min
        return self.quanData
