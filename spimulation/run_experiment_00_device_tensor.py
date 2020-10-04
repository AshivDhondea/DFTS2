"""
run_experiment_00_device_tensor.py

Called by main_00_generate_tensors.py

Splits a given Keras model at the chosen split layer.
For the device sub-model, generates the output tensors.
Saves these tensors in npy files.

Based on test_config.py from the original DFTS.

Changelog:

"""
import sys
sys.path.append('..')

import numpy as np
import os
import tensorflow as tf

from .utils import *

from models.BrokenModel import BrokenModel as BM
from .simmods import *

from models.packetModel import PacketModel as PacketModel
# --------------------------------------------------------------------------- #

def runSimulation(model, splitLayer, task, modelDict, simDir, customObjects, evaluator):
    """
    Run a simulation experiment.

    Parameters
    ----------
    model : tf.keras model or path to a tf.keras model
        Tensorflow.keras model or path to it, along with its architecture and weights.
    splitLayer : string
        Name of layer at which the DNN model should be split.
    task : 0 or 1 (bool)
        Two types of task: classification (0) and object detection (1).
    modelDict : dictionary
        Dictionary of models provided natively by tf.keras
    simDir : path to a directory
        Path to an existing directory to save the simulation results.
    customObjects : TYPE
        DESCRIPTION.
    evaluator : object
        Evaluation allocator object.

    Returns
    -------
    None.

    """
    # Object for data generator.
    dataGen = task.dataFlow()

    # Load the tf.keras model.
    loaded_model = modelLoader(model, modelDict, customObjects)

    # Object for splitting a tf.keras model into a mobile sub-model and a cloud
    # sub-model at the chosen split layer 'splitLayer'.
    testModel = BM(loaded_model, splitLayer, customObjects)
    testModel.splitModel()

    loaded_model_config = loaded_model.get_config()
    loaded_model_name = loaded_model_config['name']

    os.makedirs(os.path.join(simDir,loaded_model_name,splitLayer),exist_ok=True)

    label_list = []
    batch_index_list = []

    batch_label_dict = {}

    while not dataGen.runThrough:
        bI = dataGen.batch_index/dataGen.batch_size
        print(f"Batch index:{int(bI)}")
        label, data = dataGen.getNextBatch()
        print('Labels for this batch are')
        print(label)

        # --------------------------------------------------------------- #
        # Push the data through the device sub-model
        deviceOut = deviceSim(testModel.deviceModel, data)
        devOut = []
        if not isinstance(deviceOut, list):
            devOut.append(deviceOut)
            deviceOut = devOut
        # deviceOut is the output tensor for a batch of data.
        # --------------------------------------------------------------- #
        print('The shape of the tensor output by the device model is')
        print(np.shape(deviceOut))
        np.save(os.path.join(simDir,loaded_model_name,splitLayer,'device_tensor_batch_'+str(int(bI))+'.npy'),deviceOut)
        np.save(os.path.join(simDir,loaded_model_name,splitLayer,'true_labels_batch_'+str(int(bI))+'.npy'),label)

    # ------------------------------------------------------------------- #
    dataGen.runThrough = False
    evaluator.runThrough = True
    evaluator.reset()
    # ------------------------------------------------------------------- #
