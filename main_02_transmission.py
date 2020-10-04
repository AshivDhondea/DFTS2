"""
main_01_transmission.py

follows main_00_generate_tensors.py

Simulate the transmission of packets through a channel.

1. Load a batch of tensor data out of the device model.
2. Quantize the data according to the chosen quantization level.
3. Packetize the data.
4. Transmit the data through the channel.
5. Save the transmitted packets.

From terminal, run
python main_01_transmission.py -p params_01_transmission.yml

Date: Thursday October 1, 2020.
"""
# ---------------------------------------------------------------------------- #
import argparse
import re
import yaml
import sys
import os
import importlib

from spimulation.evalloc import evalAllocater
from spimulation.run_experiment_02_transmission import runSimulation
# ---------------------------------------------------------------------------- #
# class ParserError(Exception):
#     """
#     Used to throw exceptions when multiple options are selected by user.
#
#     Unchanged from original DFTS.
#
#     """
#
#     def __init__(self, message, errors):
#         """
#         Error handling.
#
#         Parameters
#         ----------
#         message : TYPE
#             DESCRIPTION.
#         errors : TYPE
#             DESCRIPTION.
#
#         Returns
#         -------
#         None.
#
#         """
#         super().__init__(message)
#         self.errors = errors
#
# def isURL(s):
#     """
#     Check if the given string is a valid http/https url.
#
#     Parameters
#     ----------
#     s: Input string, can be a url
#
#     Returns
#     -------
#     bool value stating whether the given string is a url
#     """
#     url = re.compile("""http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|
#                         (?:%[0-9a-fA-F][0-9a-fA-F]))+""")
#     return bool(url.match(s))
#
def selectParamConfig(p, paramDict):
    """
    Throw everything except for the user selected parameter.

    Parameters
    ----------
    p: parameter for which selection is being made
    paramDict: dictionary containing user selected parameters

    Returns
    -------
    Selected parameter and the corresponding values

    Raises
    -------
    ParserError: if more than one option is selected for a parameter
    """
    sum = 0
    index = 0

    for i in paramDict:
        sum += paramDict[i]['include']
        if paramDict[i]['include']:
            index = i
    try:
        if sum>1:
            raise ParserError("Multiple configurations selected for {}".format(p), sum)
    except Exception as e:
        raise
    else:
        if sum==0:
            return (index, False)
        else:
            return (index, paramDict[index])

def configSettings(config):
    """
    Refine the parameter dictionary to only include user selected parameters.

    Parameters
    ----------
    config: dictionary read from the YAML file

    Returns
    ----------
    Dictionary containing only the options selected by the user
    """
    for i in config:
        if i=='Transmission':
            for j in config[i]:
                transDict = {}
                if j=='channel':
                    index, temp = selectParamConfig(j, config[i][j])
                    transDict[index] = temp
                    config[i][j] = transDict
    return config
#
# def customImport(modules, classes, functions):
#     """
#     Import modules, classes and functions.
#
#     Parameters
#     ----------
#     modules : TYPE
#         DESCRIPTION.
#     classes : TYPE
#         DESCRIPTION.
#     functions : TYPE
#         DESCRIPTION.
#
#     Returns
#     -------
#     custDict : TYPE
#         DESCRIPTION.
#
#     """
#     custDict = {}
#     for i in range(len(modules)):
#         module = importlib.import_module(modules[i])
#         if functions[i]:
#             myClass = getattr(module, classes[i])()
#             myClass = getattr(myClass, functions[i])
#             custDict[functions[i]] = myClass
#             continue
#
#         myClass = getattr(module, classes[i])
#         custDict[classes[i]] = myClass
#         return custDict


def userInterface():
    """
    Read in the YAML config file and call the corresponding functions.

    Called by main function.

    Returns
    -------
    None.

    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--params", help="path to the config file containing parameters", required=True)

    args = vars(ap.parse_args())

    fileName = args['params']

    with open(fileName) as c:
        config = yaml.load(c,yaml.SafeLoader) # added SafeLoader. 4 June 2020
    paramsDict = configSettings(config)

    ############################################
    #Keras model params
    ############################################
    loaded_model_name  = paramsDict['Model']['kerasmodelname']
    ############################################
    # Experiment details
    ############################################
    experiment_params_dict = paramsDict['ExperimentDetails']
    ############################################
    # Tensor completion
    ############################################
    tensor_completion_dict = paramsDict['TensorCompletion']
    #############################################
    # Split layer
    #############################################
    splitLayer = paramsDict['SplitLayer']['split']
    #############################################
    # Transmission parameters
    #############################################
    transDict  = paramsDict['Transmission']

    simDir = paramsDict['InputDir']['simDataDir']

    runSimulation(loaded_model_name,experiment_params_dict, splitLayer,transDict,tensor_completion_dict, simDir)


# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    userInterface()
