import numpy as np
import collections

def NNInterp(tensor, receivedIndices, lostIndices, rowsPerPacket):
    """Performs nearest neighbour interpolation based packet loss concealment

    # Arguments
        tensor: packets to be interpolated, 5D
        receivedIndices: packets that were retained
        lostIndices: packets that were lost
        rowsPerPacket: number of rows of the feature map to be considered as one packet

    # Returns
        5D tensor whose packets have undergone loss concealment
    """
    nearestNeighDict   = nearestNeighbours(receivedIndices, lostIndices, tensor.shape)

    for i in nearestNeighDict:
        x, y = i
        if len(nearestNeighDict[i].shape) == 1:
            pck = nearestNeighDict[i][1]
            if y[1]==0:
                upperNeigh = tensor[x[0], x[1], -1, :, x[2]]
                tensor[x[0], pck, :, :, x[2]] = (np.outer(np.ones(rowsPerPacket), upperNeigh)).reshape(tensor[x[0], pck, :, :, x[2]].shape)
                continue
            if x[1]==-1:
                lowerNeigh = tensor[y[0], y[1], 0, :, y[2]]
                tensor[x[0], pck, :, :, x[2]] = (np.outer(np.ones(rowsPerPacket), lowerNeigh)).reshape(tensor[x[0], pck, :, :, x[2]].shape)
                continue
            lowerNeigh = tensor[y[0], y[1], 0, :, y[2]]
            upperNeigh = tensor[x[0], x[1], -1, :, x[2]]
            upperCoeff = np.ones(rowsPerPacket)
            upperCoeff[int(rowsPerPacket/2) :] = 0
            lowerCoeff = np.ones(rowsPerPacket)
            lowerCoeff[:int(rowsPerPacket/2)] = 0
            tensor[x[0], pck:pck+1, :, :, x[2]] = (np.outer(lowerCoeff, lowerNeigh) + np.outer(upperCoeff, upperNeigh)).reshape(tensor[x[0], pck:pck+1, :, :, x[2]].shape)
            continue
        else:
            pck = nearestNeighDict[i][:, 1]
            if y[1]==0:
                upperNeigh = tensor[x[0], x[1], -1, :, x[2]]
                upperCoeff = np.ones((tensor.shape[1]-pck[0])*rowsPerPacket)
                tensor[x[0], pck[0]:, :, :, x[2]] = (np.outer(upperCoeff, upperNeigh)).reshape(tensor[x[0], pck[0]:, :, :, x[2]].shape)
                continue
            if x[1]==-1:
                lowerNeigh = tensor[y[0], y[1], 0, :, y[2]]
                lowerCoeff = np.ones((pck[-1]+1)*rowsPerPacket)
                tensor[x[0], :pck[-1]+1, :, :, x[2]] = (np.outer(lowerCoeff, lowerNeigh)).reshape(tensor[x[0], :pck[-1]+1, :, :, x[2]].shape)
                continue
            upperNeigh = tensor[x[0], x[1], -1, :, x[2]]
            lowerNeigh = tensor[y[0], y[1], 0, :, y[2]]
            upperCoeff = np.ones((pck[-1]-pck[0]+1)*rowsPerPacket)
            upperCoeff[int(upperCoeff.shape[0]/2):] = 0
            lowerCoeff = np.ones((pck[-1]-pck[0]+1)*rowsPerPacket)
            lowerCoeff[:int(lowerCoeff.shape[0]/2)] = 0
            tensor[x[0], pck[0]:pck[-1]+1, :, :, x[2]] = (np.outer(upperCoeff, upperNeigh)+np.outer(lowerCoeff, lowerNeigh)).reshape(tensor[x[0], pck[0]:pck[-1]+1, :, :, x[2]].shape)
            continue
    return tensor

def createNeighDict(rP, lP, b, c):
    """Finds the neighbours nearest to a lost packet in a particular tensor plane

    # Arguments
        rP: packets received in that tensor plane
        lp: packets lost in that tensor plane
        b,c : batch and channel number denoting the tensor plane

    # Returns
        Dictionary containing the neighbours nearest to the lost packets
    """
    insertPos = np.searchsorted(rP, lP)
    neighDict = collections.OrderedDict()

    if len(rP)==0:
        return neighDict

    for i in range(len(lP)):
        ind = insertPos[i] #position at which lP is to be inserted in rP
        if ind==0: #check if insert position is at beginning i.e no top neighbour
            k = ((b, -1, c), (b, rP[ind], c))
        # k = (tuple((b, -1, c)), tuple((b, rP[ind], c)))
            v = np.array([b, lP[i], c])
            if k not in neighDict:
                neighDict[k] = v
            else:
                neighDict[k] = np.vstack((neighDict[k], v))
            continue
        if ind==len(rP): #check if insert position is at the end i.e no bottom neighbour
            k = ((b, rP[-1], c), (b, 0, c))
            # k = (tuple((b, rP[-1], c)), tuple((b, 0, c)))
            v = np.array([b, lP[i], c])
            if k not in neighDict:
                neighDict[k] = v
            else:
                neighDict[k] = np.vstack((neighDict[k], v))
            continue
        k = ((b, rP[ind-1], c), (b, rP[ind], c))
        # k = (tuple((b, rP[ind-1], c)), tuple((b, rP[ind], c)))
        v = np.array([b, lP[i], c])
        if tuple(k) not in neighDict:
            neighDict[k] = v
        else:
            neighDict[k] = np.vstack((neighDict[k], v))
    return neighDict


def nearestNeighbours(receivedIndices, lostIndices, tensorShape):
    """Creates a complete dictionary containing lost indices and their nearest neighbours

    # Arguments
        receivedIndices: packets that were retained
        lostIndices: packets that were lost
        tensorShape: tuple denoting the shape of 5D tensor of packets

    # Returns
        Dictionary of nearest neighBours
    """
    nearestNeighDict = {}

    #each value in the receivedIndices and lossIndices is a 3d tuple
    bS          = tensorShape[0]
    ch          = tensorShape[-1]
    bS          = np.arange(0, bS, 1)
    ch          = np.arange(0, ch, 1)
    xx, yy   = np.meshgrid(bS, ch)
    bChPairs = np.dstack((xx, yy)).reshape(-1, 2)

    #can speed up following using multiprocessing

    for b,c in bChPairs:
        rInds = np.where(np.logical_and(receivedIndices[:, :, 0]==b, receivedIndices[:, :, 2]==c))
        lInds = np.where(np.logical_and(lostIndices[:, :, 0]==b, lostIndices[:, :, 2]==c))
        nearestNeighDict.update(createNeighDict(receivedIndices[rInds[:][0], rInds[:][1], 1], lostIndices[lInds[:][0], lInds[:][1], 1], b, c))
    return nearestNeighDict
