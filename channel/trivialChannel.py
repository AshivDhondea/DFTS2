"""
Trivial channel class.

Random loss channel model.

Unchanged from original DFTS.
"""
import numpy as np

class RLC(object):
    """Simulate a random loss channel."""
    
    def __init__(self, lossProb):
        """
        Initialize a random loss channel model class.
        
        # Arguments
            lossProb: loss probability of the channel
        """
        super(RLC, self).__init__()
        self.lossProb   = lossProb
        self.lossMatrix = []

    def simulate(self, lossSize):
        """
        Define the packets that are lost.
        
        # Arguments
            lossSize: number of packets
        # Returns
            A matrix containing values that are lost and retained.
        """
        self.lossMatrix = np.random.random(lossSize)
        probMatrix      = np.full(lossSize, self.lossProb)

        return np.greater_equal(self.lossMatrix, probMatrix).astype('float64')
