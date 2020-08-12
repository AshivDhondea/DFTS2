"""
Split the tf.Keras model into a mobile sub-model and a cloud sub-model.

# Changelog:
    7 August 2020: Fixed commenting and docstrings.
"""
#import keras hans commented out 17 june 2020.
#import tensorflow as tf
#from tf.keras.layers import Input
#from tf.keras.models import Model

# 22 June 2020
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

# August 6 2020
import tensorflow as tf

from .utils.cloud import remoteModel, modelOut
#import numpy as np # imported but unused.

class BrokenModel(object):
    """
    Split the model at the given layer into mobile sub-model and cloud sub-model.
    
    Unchanged from original DFTS.
    
    """
    
    def __init__(self, model, splitLayer, custom_objects):
        """
         Initialize a BrokenModel class object.

        Parameters
        ----------
        model : tf.keras model.
            This model represents the full trained DNN, including weights.
        splitLayer : string 
            A string representing the layer at which the model needs to be split.
        custom_objects : TYPE
            DESCRIPTION. Hans: unused up to now.

        Returns
        -------
        None.
        """
        super(BrokenModel, self).__init__()
        self.model      = model
        self.layers     = [i.name for i in self.model.layers]
        self.splitLayer = splitLayer
        self.layerLoc   = self.layers.index(self.splitLayer)
        self.custom_objects = custom_objects

    def splitModel(self):
        """
        Split the tf.keras model into the device model (on the edge/mobile device) and the cloud model (remote model) at the specified layer.
        
        Returns
        -------
        None.
        """
        # modelOut returns         
        deviceOuts, remoteIns, skipNames = modelOut(self.model, self.layers, self.layerLoc)

        self.deviceModel = tf.keras.models.Model(inputs=self.model.input, outputs=deviceOuts)
        self.remoteModel = remoteModel(self.model, self.splitLayer, self.custom_objects)
