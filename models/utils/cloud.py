"""
Functions related to BrokenModel.

edited: 6 August 2020.

# Changelog:
    17 June 2020: Changed the imports, now importing tensorflow directly.
    6 August 2020: Adapted remoteModel to run in TFv2.
    28 September 2020: added code to rename cloud model and delete unconnected input layers.
    5 February 2021: cleaned up and added descriptions.
"""
import tensorflow as tf
#import copy
# ---------------------------------------------------------------------------------------- #
def modelOut(loaded_model, layers, index):
    """
    Find the output tensor of the mobile sub-model and the input tensor for the cloud sub-model if the <loaded_model> (with layer names in <layers>) is split at the layer given by     <index>.

    Adapted from the original DFTS code for TFv2 compatibility.

    # Arguments
        model: keras model
        layers: list of strings representing the names of the layer in the model.
        index: location of the layer where the model is split.

    # Returns
        Ouput tensor of the device model, inputs of the remote model, strings representing the
        names of the layers to be skipped.
    """
    device = set(layers[:index+1])
    remote = layers[index+1:]

    deviceOuts = []
    remoteIns  = []
    skipNames  = []

    for i in remote:
        rIndex = layers.index(i)
        layer_input_tensor= loaded_model.layers[rIndex].input
        if type(layer_input_tensor) is list: # Model contains skip connections.
           for i_lower in range(len(layer_input_tensor)):
               # loop through elements in list layer_input_tensor
               lower_layer = layer_input_tensor[i_lower]

               for j in device: # Go through layers and if the input of a layer corresponds to the output of another layer, then they are connected.
                   dIndex = layers.index(j)
                   layer_output_tensor = loaded_model.layers[dIndex].output
                   if lower_layer.name == layer_output_tensor.name and lower_layer.shape == layer_output_tensor.shape and lower_layer.dtype == layer_output_tensor.dtype:
                       r = tf.keras.layers.Input(layer_output_tensor.shape[1:])
                       deviceOuts.append(layer_output_tensor)
                       remoteIns.append(r)
                       skipNames.append(loaded_model.layers[dIndex].name)
        else: # Model does not contain skip connections.
            for j in device:  # Go through layers and if the input of a layer corresponds to the output of another layer, then they are connected.
                dIndex = layers.index(j)
                layer_output_tensor = loaded_model.layers[dIndex].output

                if layer_input_tensor.name == layer_output_tensor.name and layer_input_tensor.shape == layer_output_tensor.shape and layer_input_tensor.dtype == layer_output_tensor.dtype:
                    r = tf.keras.layers.Input(layer_output_tensor.shape[1:])
                    deviceOuts.append(layer_output_tensor)
                    remoteIns.append(r)
                    skipNames.append(loaded_model.layers[dIndex].name)

    return deviceOuts, remoteIns, skipNames
# ---------------------------------------------------------------------------------------- #
def createInpCfg(inp):
    """
    Create input layer configuration.

    Parameters
    ----------
    inp : Input layer.

    Returns
    -------
    cfg : Dictionary representing the configuration of the layer.

    """
    cfg = {}
    cfg['name'] = inp.name.split(':')[0]
    cfg['class_name'] = 'InputLayer'
    cfg['config'] = {'batch_input_shape':tuple(inp.shape.as_list()),
    'dtype':'float32', 'sparse':False, 'name':inp.name.split(':')[0]}
    cfg['inbound_nodes'] = []

    return cfg

def createRMCfg(loaded_model, remoteIns, deviceOuts, index):
    """Create the remote model's configuration dictionary.

    # Arguments
        loaded_model: keras model for the full DNN.
        remoteIns: input tensors to the remote model
        deviceOuts: output tensors from the device model
        index: location of the layer where the model is split

    # Returns
        Dictionary representing the configuration of the remote model
    """
    deviceOuts = [i.name for i in deviceOuts]
    modelCfg = loaded_model.get_config()

    remoteIns = [createInpCfg(i) for i in remoteIns]

    modelLayers = modelCfg['layers'][index+1:]

    for i in remoteIns:
        modelLayers.insert(0, i)
    modelCfg['layers'] = modelLayers

    return modelCfg

def fn_set_weights(smaller_model,original_model):
    """
    Reset the weights for the smaller tf.keras model created.

    Parameters
    ----------
    smaller_model : tf.keras model
        Model for the smaller model obtained from the full model.
    original_model : tf.keras model
        Model for the original model (the full model).

    Returns
    -------
    smaller_model : tf.keras model
       Model for the smaller model with the weights properly set.

    """
    modelLayers = [i.name for i in original_model.layers]
    for l in smaller_model.layers:
        orig = l.name
        if orig in modelLayers:
            lWeights = original_model.get_layer(orig)
            l.set_weights(lWeights.get_weights())
    return smaller_model

def remoteModel(loaded_model,split_layer,custom_objects=None):
    """
    Create the remote sub-model <cloud_model> for the <loaded_model> split at layer <split>.

    Based on:
        https://stackoverflow.com/questions/49193510/how-to-split-a-model-trained-in-keras

    Parameters
    ----------
    loaded_model : tf.keras model
        Full tf.keras model which needs to be split.
    split_layer : string
        String representing the split layer in the DNN model.
    custom_objects : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    cloud_model : tf.keras model
        tf.keras model representing the cloud sub-model, including weights.
        Can be used for inference.

    """
    layers = [i.name for i in loaded_model.layers]
    layerLoc = layers.index(split_layer)

    deviceOuts, remoteIns, skipNames =modelOut(loaded_model, layers,layerLoc)

    # names for layers used for input to the remote model
    inNames = [i.name.split(':')[0] for i in remoteIns]

    remote_config = createRMCfg(loaded_model, remoteIns, deviceOuts,layerLoc)

    for i in remote_config['layers']:
        if len(i['inbound_nodes'])==0:
            # Continue for input layers to the remote model.
            continue
        # Sort out inbound nodes for layers downstream of the input layer(s) of
        # remote model.
        temp = i['inbound_nodes'][0]
        for j in temp:
            if j[0] in skipNames:
                jIndex = temp.index(j)
                j[0] = inNames[skipNames.index(j[0])]
                temp[jIndex] = j
        i['inbound_nodes'][0] = temp

    remote_config['input_layers'] = []
    for i in inNames:
        # Add input layers' dictionary to configuration dictionary.
        remote_config['input_layers'].append([i, 0, 0])

    # Set the name of the cloud model.
    remote_config['name'] = 'remote_sub_model'

    cloud_model = tf.keras.Model.from_config(remote_config,custom_objects=custom_objects)
    cloud_model = fn_set_weights(cloud_model,loaded_model)
    return cloud_model
