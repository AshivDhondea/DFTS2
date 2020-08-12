"""

Simulator class for two types of task: classification and object detection.

"""
import os
import numpy as np
from .utils import absoluteFilePaths
from .dataGen import CFDataGenerator as CFDG
from .dataGen import ODDataGenerator as ODDG

class CFTask(object):
    """Initializes the simulator for classfication.
    """
    def __init__(self, testDir, reshapeDims, batch_size=64, normalize=False):
        """Initialize the class and gather the test data

        # Arguments
            testDir: top level directory containing the images
            batch_size: number of images to be forwarded through the model at once
            reshapeDims: list containing the dimensions of the reshaped image
            normalize: bool value indicating whether images must be normalized
        """
        super(CFTask, self).__init__()
        self.testDir     = testDir
        self.batch_size  = batch_size
        self.reshapeDims = reshapeDims
        self.normalize   = normalize
        self.gatherData()

    def gatherData(self):
        """Organizes the data into an array containing the labels and path to the images.
        """
        dirList = os.listdir(self.testDir)
        self.numClasses = len(dirList)
        dirList = [os.path.join(self.testDir, i).replace("\\", "/") for i in dirList]
        images  = [absoluteFilePaths(i) for i in dirList]

        classImgArr = []

        for i in range(len(images)):
            exp = [list((int(dirList[i].split('/')[-1]), images[i][j])) for j in range(len(images[i]))]
            classImgArr.append(exp)

        classImgArr = np.array([item for sublist in classImgArr for item in sublist])
        labels = classImgArr[:, 0].astype(np.int32)
        # labelsTest = np.array([i.split('\\')[-2] for i in classImgArr[:, 1]])
        classImgArr[:, 0] = labels
        self.testData = classImgArr

    def dataFlow(self):
        """Create a data generator based on the given parameters

        # Returns
            A Data Generator object.
        """
        params = {
            'reshapeDims': self.reshapeDims,
            'batch_size' : self.batch_size,
            'n_classes'  : self.numClasses,
            'normalize'  : self.normalize
        }
        return CFDG(self.testData, **params)


class ODTask(object):
    """Initializes the simulator for object detection"""
    def __init__(self, testDir, reshapeDims, batch_size, classes, num_classes):
        """
        # Arguments
            testDir: dictionary conatining the annotations, images and file containing the list of images
            batch_size: number of images to be forwarded through the model at once
            reshapeDims: list containing the dimensions of the reshaped image
            normalize: bool value indicating whether images must be normalized
        """
        super(ODTask, self).__init__()
        self.testDir     = testDir
        self.reshapeDims = reshapeDims
        self.batch_size  = batch_size
        self.classes     = classes
        self.num_classes = num_classes

        #move parse data function to this class to handle datasets other than PASCAL VOC

    def dataFlow(self):
        """Create a data generator based on the given parameters

        # Returns
            A Data Generator object.
        """
        return ODDG(self.testDir, self.reshapeDims, self.batch_size, self.classes, self.num_classes)
