"""
Simulation module functions.

Documentation to be sorted out later.

"""
import sys
sys.path.append('..')

import numpy as np
import time

from models.packetModel import PacketModel as PM


def deviceSim(model, data):
    """
    Simulates the model run on the user's device

    # Arguments
        model: keras model
        data : preprocessed data

    # Returns
        Result of the device simulation
    """
    start_time = time.time()
    deviceOut         = model.predict(data)
    total_time = time.time() - start_time
    print(f"Device simulation Complete in {total_time}!!")
    return deviceOut

# def compress(deviceOut):
#     #initially identity function
#     return deviceOut
#
def transmit(compressOut, channel, rowsPerPacket):
    """
    Simulates packetization and transmission of the packets through a channel

    # Arguments
        compressOut: TODO
        channel: channel object
        rowsPerPacket: number of rows of the feature map to be considered as one packet

    # Returns
        Packetized and lost data along with the indices of the lost and retained packets
    """
    start_time   = time.time()
    pckts        = PM(compressOut, rowsPerPacket)

    lossMatrix = channel.simulate(pckts.packetSeq.shape[0]*pckts.packetSeq.shape[1]*pckts.packetSeq.shape[-1])
    lossMatrix = lossMatrix.reshape(pckts.packetSeq.shape[0], pckts.packetSeq.shape[1], pckts.packetSeq.shape[-1])

    receivedIndices = np.where(lossMatrix==1)
    receivedIndices = np.dstack((receivedIndices[0], receivedIndices[1], receivedIndices[2]))

    lostIndices = np.where(lossMatrix==0)
    lostIndices = np.dstack((lostIndices[0], lostIndices[1], lostIndices[2]))

    pckts.packetSeq[lostIndices[:,:,0], lostIndices[:,:,1], :, :, lostIndices[:,:,-1]] = 0

    total_time = time.time() - start_time
    print(f"Transmission Complete in {total_time}!!")
    return (pckts, lossMatrix, receivedIndices, lostIndices)

def errorConceal(interpPackets, pBuffer, receivedIndices, lostIndices, rowsPerPacket):
    """Performs packet loss concealment on the given data.

    # Arguments
        interpPackets: function object corresponding to a particular interpolation kind
        pBuffer: packets to be interpolated
        receivedIndices: packets that were retained
        lostIndices: packets that were lost
        rowsPerPacket: number of rows of the feature map to be considered as one packet

    # Returns
        Tensor whose loss has been concealed
    """
    print("Error Concealment")
    return interpPackets(pBuffer, receivedIndices, lostIndices, rowsPerPacket)

def remoteSim(remoteModel ,channelOut, channel):
    """Simulates the model that is run on the cloud.

    # Arguments
        remoteModel: keras model in the cloud
        channelOut : packets of data

    # Returns
        Prections for a particular batch of images
    """
    if channel=='noChannel':
        cOut = []
        for i in range(len(channelOut)):
            cOut.append(channelOut[i])
        return remoteModel.predict(cOut)
    cOut = []
    for i in range(len(channelOut)):
        data = channelOut[i].packetToData()
        cOut.append(data)
    return remoteModel.predict(cOut)
