"""
Functions related to BrokenModel.

edited: 6 August 2020.

# Changelog:
    17 June 2020: Changed the imports, now importing tensorflow directly.
    6 August 2020: Adapted remoteModel to run in TFv2.
                   Removed outdated functions createInpCfg and createRMCfg.

"""
import tensorflow as tf

def modelOut(model, layers, index):
    """
    Produce the outputs of the model on the device.

    # Arguments
        model: keras model
        layers: list of strings representing the names of the layer in the model
        index: location of the layer where the model is split

    # Returns
        Ouputs of the device model, inputs of the remote model, strings representing the 
        names of the layers to be skipped
    """
    device = set(layers[:index+1])
    remote = layers[index+1:]

    deviceOuts = []
    remoteIns  = []
    skipNames  = []

    for i in remote:
        rIndex = layers.index(i)
        #curIn = model.layers[rIndex].input
        # Hans. 21 June 2020
        layer_input_tensor= model.layers[rIndex].input
        for j in device:
            dIndex = layers.index(j)
            #out = model.layers[dIndex].output
            layer_output_tensor = model.layers[dIndex].output
            if layer_input_tensor.name == layer_output_tensor.name and layer_input_tensor.shape == layer_output_tensor.shape and layer_input_tensor.dtype == layer_output_tensor.dtype: #curIn==out:
                 #d = model.layers[index].output
                r = tf.keras.layers.Input(layer_output_tensor.shape[1:])#(out.shape[1:])
                deviceOuts.append(layer_output_tensor)#(out)
                remoteIns.append(r)
                skipNames.append(model.layers[dIndex].name)

    return deviceOuts, remoteIns, skipNames

def remoteModel(loaded_model,split,custom_objects=None):
    """
    Implement the remote sub-model for BrokenModel.
    
    Based on:
        https://stackoverflow.com/questions/49193510/how-to-split-a-model-trained-in-keras

    Parameters
    ----------
    loaded_model : tf.keras model
        Full tf.keras model which needs to be split.
    split : string
        String representing the split layer in the DNN model.
    custom_objects : TYPE, optional
        DESCRIPTION. The default is None.
        Hans: unused up to now.

    Returns
    -------
    cloud_model : tf.keras model
        tf.keras model representing the cloud sub-model, including weights.
        Can be used for inference.

    """
    layers = [i.name for i in loaded_model.layers]
    layerLoc = layers.index(split)  
    
    cloud_input = tf.keras.layers.Input(loaded_model.layers[layerLoc+1].input_shape[1:])
    cloud_model = cloud_input
    for layer in loaded_model.layers[layerLoc+1:]:
        cloud_model = layer(cloud_model)
    cloud_model = tf.keras.models.Model(inputs=cloud_input, outputs=cloud_model)
    
    return cloud_model