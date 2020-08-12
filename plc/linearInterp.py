import numpy as np
import collections

def interpPackets(tensor, receivedIndices, lostIndices, rowsPerPacket):
    """Performs linear interpolation based packet loss concealment

    # Arguments
        tensor: packets to be interpolated, 5D
        receivedIndices: packets that were retained
        lostIndices: packets that were lost
        rowsPerPacket: number of rows of the feature map to be considered as one packet

    # Returns
        5D tensor whose packets have undergone loss concealment
    """
    nearestNeighDict   = nearestNeighbours(receivedIndices, lostIndices, tensor.shape)
    coeff              = interpCoeff(nearestNeighDict, rowsPerPacket, tensor.shape)

    j = 0
    for i in nearestNeighDict:
        #i is the neighbours
        x,y = i
        if len(nearestNeighDict[i].shape) == 1:
            #only one lost packet between the neighbours
            pck = nearestNeighDict[i][1]
            if y[1]==0: #packets lost at the end
                lowerNeigh = tensor[y[0], y[1], 0, :, y[2]]
                upperNeigh = tensor[x[0], x[1], -1, :, x[2]]
                tensor[x[0], pck:, :, :, x[2]] =(np.outer(coeff[j][0], lowerNeigh)
                                                +np.outer(coeff[j][1], upperNeigh)).reshape(tensor[x[0], pck:, :, :, x[2]].shape)
                j = j+1
                continue
            if x[1]==-1: #packets lost in the beginning
                lowerNeigh = tensor[y[0], y[1], 0, :, y[2]]
                upperNeigh = tensor[x[0], x[1], -1, :, x[2]]
                tensor[x[0], :pck+1, :, :, x[2]] =(np.outer(coeff[j][0], lowerNeigh)
                                                +np.outer(coeff[j][1], upperNeigh)).reshape(tensor[x[0], :pck+1, :, :, x[2]].shape)
                j = j+1
                continue
            lowerNeigh = tensor[y[0], y[1], 0, :, y[2]]
            upperNeigh = tensor[x[0], x[1], -1, :, x[2]]
            tensor[x[0], pck:pck+1, :, :, x[2]] =(np.outer(coeff[j][0], lowerNeigh)
                                            +np.outer(coeff[j][1], upperNeigh)).reshape(tensor[x[0], pck:pck+1, :, :, x[2]].shape)
            j = j+1
            continue
        else:
            pck = nearestNeighDict[i][:, 1]
            if y[1]==0: #packets lost at the end
                lowerNeigh = tensor[y[0], y[1], 0, :, y[2]]
                upperNeigh = tensor[x[0], x[1], -1, :, x[2]]
                tensor[x[0], pck[0]:, :, :, x[2]] =(np.outer(coeff[j][0], lowerNeigh)
                                                +np.outer(coeff[j][1], upperNeigh)).reshape(tensor[x[0], pck[0]:, :, :, x[2]].shape)
                j = j+1
                continue
            if x[1]==-1: #packets lost in the beginning
                lowerNeigh = tensor[y[0], y[1], 0, :, y[2]]
                upperNeigh = tensor[x[0], x[1], -1, :, x[2]]
                tensor[x[0], :pck[-1]+1, :, :, x[2]] =(np.outer(coeff[j][0], lowerNeigh)
                                                +np.outer(coeff[j][1], upperNeigh)).reshape(tensor[x[0], :pck[-1]+1, :, :, x[2]].shape)
                j = j+1
                continue
            lowerNeigh = tensor[y[0], y[1], 0, :, y[2]]
            upperNeigh = tensor[x[0], x[1], -1, :, x[2]]
            tensor[x[0], pck[0]:pck[-1]+1, :, :, x[2]] =(np.outer(coeff[j][0], lowerNeigh)
                                            +np.outer(coeff[j][1], upperNeigh)).reshape(tensor[x[0], pck[0]:pck[-1]+1, :, :, x[2]].shape)
            j = j+1 #damn, this bug!!
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


def interpCoeff(neighBours, rowsPerPacket, tensorShape):
    """Generates the coefficients for interpolation

    # Arguments
        neighBours: dictionary containing lost indices and their nearest neighbours
        rowsPerPacket: number of rows of the feature map to be considered as one packet
        tensorShape: tuple denoting the shape of 5D tensor of packets

    # Returns
        List containing the interpolation coefficients
    """
    coeff = []
    for (x, y) in neighBours:
        if x[1]==-1:
            xS   = y[1]
            aVec = np.ones((1, xS*rowsPerPacket)) #lower neighbour
            bVec = 1-aVec #upper neighbour
            coeff.append(np.vstack((aVec, bVec)))
            # print(np.vstack((aVec, bVec)).shape)
        elif y[1]==0:
            yS   = tensorShape[1]-x[1]-1
            bVec = np.ones((1, rowsPerPacket*yS)) #upper neighbour
            aVec = 1-bVec #lower neighbour
            coeff.append(np.vstack((aVec, bVec)))
            # print(np.vstack((aVec, bVec)).shape)
        else:
            aVec = np.arange(1, (y[1]-x[1]-1)*rowsPerPacket + 1)/(((y[1]-x[1]-1)*rowsPerPacket+1))
            bVec = 1-aVec
            coeff.append(np.vstack((aVec, bVec)))
            # print(np.vstack((aVec, bVec)).shape)
    return coeff
