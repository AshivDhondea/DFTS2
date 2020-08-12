"""
Set up configurations for the experiment.
 
Documentation to be sorted out later.

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
    """Runs a simulation based on the given parameters.

    Forwaards the data through the model on the device, transmits it, forwards it through the model
    on the cloud and then generates predictions.
    """
    dataGen = task.dataFlow()

    model = modelLoader(model, modelDict, customObjects)

    testModel = BM(model, splitLayer, customObjects)

    testModel.splitModel()

    rowsPerPacket = transDict['rowsperpacket']
    quantization  = transDict['quantization']
    channel       = transDict['channel']
    lossConceal   = transDict['concealment']

    channel = loadChannel(channel)
    quant   = quantInit(quantization)
    conceal = plcLoader(lossConceal)

    fileName = createFile(quant, conceal, splitLayer)
    fileName = os.path.join(simDir, fileName)

    testData = []
    userRes = []

    for e in range(epochs):
        print("Epoch number:{}".format(e))
        while not dataGen.runThrough:
            quanParams = []
            bI = dataGen.batch_index/dataGen.batch_size
            print("Batch number:{}".format(bI))
            label, data = dataGen.getNextBatch()
            # print(dataGen.batch_index)
            deviceOut = deviceSim(testModel.deviceModel, data)
            devOut = []
            if not isinstance(deviceOut, list):
                devOut.append(deviceOut)
                deviceOut = devOut

            if quant!='noQuant':
                for i in range(len(deviceOut)):
                    quant.bitQuantizer(deviceOut[i])
                    deviceOut[i] = quant.quanData
                    # print(np.unique(deviceOut[i]).size)
                    quanParams.append([quant.min, quant.max])
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
            if conceal!='noConceal':
                for i in range(len(deviceOut)):
                    deviceOut[i].packetSeq = errorConceal(conceal, deviceOut[i].packetSeq, receivedIndices[i], lostIndices[i], rowsPerPacket)
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
            remoteOut = remoteSim(testModel.remoteModel, deviceOut, channel)
            evaluator.evaluate(remoteOut, label)
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

    print(userRes)
    np.save(fileName, np.array(userRes))
