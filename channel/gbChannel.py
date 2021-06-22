"""
Gilbert-Elliott channel class.

Unchanged from original DFTS.
"""

import random
import numpy as np

class GBC(object):
    """Simulate a Gilbert-Elliott channel."""
    
    def __init__(self, lossProb, burstLength):
        """
        Initialize a Gilbert-Elliott channel model class.
        
        # Arguments
            lossProb: probability of packet loss
            burstLength: burst of loss
        """
        super(GBC, self).__init__()
        self.lp = lossProb #fixed initially
        self.bl = burstLength
        self.state = True#1 #initially bad channel state
        self.calcChannelProb()
        self.lossMatrix = []

    def calcChannelProb(self):
        """Calculate the probabilities of transition from good to bad channel and vice versa."""
        self.pbg = 1.0/self.bl
        self.pgb = self.pbg/((1.0/self.lp)-1)

    def simulate(self, nofSims):
        """
        Create a series of good and bad channel states.

        # Returns
            array containing the sequence of good and bad states
        """
        for i in range(nofSims):
            self.flip(self.state)
        return np.array(self.lossMatrix)

    def flip(self, state):
        """
        Determine if the next channel state is good or bad based on the current state.

        # Arguments
            state: current state of the channel
        """
        self.lossMatrix.append(state)
        if state==True:
            p = random.random()
            if p<self.pgb:
                self.state = False
                return
            return
        else:
            p = random.random()
            if p<self.pbg:
                self.state = True
                return
            return
