"""
Split the tf.Keras model into a mobile sub-model and a cloud sub-model.

# Changelog:
    7 August 2020: Fixed commenting and docstrings.
    15 September 2020: Modified to handle models with skip connctions.
    28 September 2090: Modified to reset the weights of the mobile and cloud models
    after creating their models from dictionaries.

"""

import tensorflow as tf

from .utils.cloud import remoteModel, modelOut, fn_set_weights
# ---------------------------------------------------------------------------------------- #
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

        device_model = tf.keras.models.Model(inputs=self.model.input, outputs=deviceOuts[0])
        device_config = device_model.get_config()
        # Set the name of the mobile model.
        device_config['name'] = 'device_sub_model'
        device_model = tf.keras.Model.from_config(device_config, custom_objects = self.custom_objects)
        self.deviceModel = fn_set_weights(device_model,self.model)
        self.remoteModel = remoteModel(self.model, self.splitLayer, self.custom_objects)
