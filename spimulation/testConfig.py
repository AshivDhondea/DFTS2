"""
Set up configurations for the experiment.

Changelog:
    12 August 2020: added comments explaining operation.


"""
import sys
sys.path.append('..')

import numpy as np
import os

from .utils import *

from models.BrokenModel import BrokenModel as BM
from .simmods import *

from .calloc import loadChannel, quantInit, plcLoader
    
def runSimulation(model, epochs, splitLayer, task, modelDict, transDict, simDir, customObjects, evaluator):
    """
    Run a simulation experiment.

    Parameters
    ----------
    model : tf.keras model or path to a tf.keras model
        Tensorflow.keras model or path to it, along with its architecture and weights.
    epochs : integer
        Number of Monte Carlo runs for the MC experiment.
    splitLayer : string
        Name of layer at which the DNN model should be split.
    task : 0 or 1 (bool)
        Two types of task: classification (0) and object detection (1).
    modelDict : dictionary
        Dictionary of models provided natively by tf.keras
    transDict : dictionary
        Dictionary containing parameters for the transmission.
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
    model = modelLoader(model, modelDict, customObjects)

    # Object for splitting a tf.keras model into a mobile sub-model and a cloud
    # sub-model at the chosen split layer 'splitLayer'.
    testModel = BM(model, splitLayer, customObjects)
    testModel.splitModel()
    
    # parameters for the transmission.
    rowsPerPacket = transDict['rowsperpacket']
    quantization  = transDict['quantization']
    channel       = transDict['channel']
    lossConceal   = transDict['concealment']

    # Objects for the channel, quantization and error concealment.
    channel = loadChannel(channel)
    quant   = quantInit(quantization)
    conceal = plcLoader(lossConceal)

    # Create results file.
    fileName = createFile(quant, conceal, splitLayer)
    fileName = os.path.join(simDir, fileName)

    testData = []
    userRes = []

    for e in range(epochs):
        # Run through for each Monte Carlo simulation.
        print("Epoch number:{}".format(e))
        while not dataGen.runThrough:
            quanParams = []
            bI = dataGen.batch_index/dataGen.batch_size
            print("Batch number:{}".format(bI))
            label, data = dataGen.getNextBatch()
            # print(dataGen.batch_index)
            # --------------------------------------------------------------- #
            # Push the data through the device sub-model
            deviceOut = deviceSim(testModel.deviceModel, data)
            devOut = []
            if not isinstance(deviceOut, list):
                devOut.append(deviceOut)
                deviceOut = devOut
            # deviceOut is the output tensor for a batch of data.
            # --------------------------------------------------------------- #
            # On the mobile side:
            # quantize the output of the device model (if needed).
            ##
            # Quantize the data
            if quant!='noQuant':
                for i in range(len(deviceOut)):
                    quant.bitQuantizer(deviceOut[i])
                    deviceOut[i] = quant.quanData
                    # print(np.unique(deviceOut[i]).size)
                    quanParams.append([quant.min, quant.max])
            ##
            # Transmit the tensor deviceOut through the channel.
            if channel!='noChannel':
                lossMatrix = []
                receivedIndices = []
                lostIndices = []
                dOut = []
                for i in range(len(deviceOut)):
                    dO, lM, rI, lI = transmit(deviceOut[i], channel, rowsPerPacket)
                    dOut.append(dO)
                    lossMatrix.append(lM)
                    receivedIndices.append(rI)
                    lostIndices.append(lI)
                    channel.lossMatrix = []
                deviceOut = dOut
            ##
            # Error concealment
            if conceal!='noConceal':
                for i in range(len(deviceOut)):
                    deviceOut[i].packetSeq = errorConceal(conceal, deviceOut[i].packetSeq, receivedIndices[i], lostIndices[i], rowsPerPacket)
                    
            # --------------------------------------------------------------- #
            # On the cloud side:
            # if the tensor was quantized, inverse quantize it.
            # if a channel was used, inverse quantize the packets.
            if quant!='noQuant':
                for i in range(len(deviceOut)):
                    if channel!='noChannel':
                        quant.quanData = deviceOut[i].packetSeq
                        qMin, qMax = quanParams[i]
                        quant.min = qMin
                        quant.max = qMax
                        deviceOut[i].packetSeq = quant.inverseQuantizer()
                    else:
                        quant.quanData = deviceOut[i]
                        qMin, qMax = quanParams[i]
                        quant.min = qMin
                        quant.max = qMax
                        deviceOut[i] = quant.inverseQuantizer()
            ## 
            # Push through the tensor through the cloud sub-model.
            remoteOut = remoteSim(testModel.remoteModel, deviceOut, channel)
            # Evaluate the results according to their label.
            evaluator.evaluate(remoteOut, label)
        # ------------------------------------------------------------------- #
        # Compile the results.
        results = evaluator.simRes()
        # print(results)
        tempRes = [e]*len(results)

        for i in range(len(results)):
            t = [tempRes[i]]
            if isinstance(results[i], list):
                [t.append(j) for j in results[i]]
            else:
                [t.append(results[i])]
            userRes.append(t)
        dataGen.runThrough = False
        evaluator.runThrough = True
        evaluator.reset()
        # ------------------------------------------------------------------- #
    # Save the results.        
    print(userRes)
    np.save(fileName, np.array(userRes))
