"""
Trivial channel class.

Documentation to be sorted out later.


"""
import numpy as np

class RLC(object):
    """Simulates a random loss channel"""
    def __init__(self, lossProb):
        """
        # Arguments
            lossProb: loss probability of the channel
        """
        super(RLC, self).__init__()
        self.lossProb   = lossProb
        self.lossMatrix = []

    def simulate(self, lossSize):
        """ Defines the packets that are lost.

        # Arguments
            lossSize: number of packets

        # Returns
            A matrix containing values that are lost and retained.
        """
        self.lossMatrix = np.random.random(lossSize)
        probMatrix      = np.full(lossSize, self.lossProb)

        return np.greater_equal(self.lossMatrix, probMatrix).astype('float64')
