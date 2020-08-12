"""
Data generator for DFTS experiment.

Documentation to be sorted out later.

Object Detection untested as of yet.

"""

import numpy as np
#from PIL import Image
import cv2
import os
import sys
from .utils import pTasks
from tqdm import tqdm
#from tqdm import trange # imported but unused.
from bs4 import BeautifulSoup
#from tensorflow.keras.preprocessing import image # hans. 17 June 2020.

# # Hans 22 June 2020
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import tensorflow as tf

class CFDataGenerator(object):
    """Generates preprocessed data in batches of given size.
    """
    
    def __init__(self, testData, **params):
        """
        # Arguments
            testData: directory location of the images
            params  : dictionary containing pre processing parameters
        """
        super(CFDataGenerator, self).__init__()
        self.testData    = testData
        self.reshapeDims = params['reshapeDims']
        self.batch_size  = params['batch_size']
        self.normalize   = params['normalize']
        self.n_classes   = params['n_classes']
        self.batch_index = 0
        self.runThrough  = False

    def getNextBatch(self):
        """Cycles through the data one batch_size at a time

        # Returns
            Current batch of preprocessed images
        """
        currentTestData = []
        if self.batch_index>=len(self.testData):
            #assume that batch_size<number of test images
            self.batch_index = 0
            currentTestData  = self.preprocess(self.testData[self.batch_index:self.batch_index+self.batch_size])
        elif self.batch_index + self.batch_size >= len(self.testData) and self.batch_index<len(self.testData):
            currentTestData  = self.preprocess(self.testData[self.batch_index:])
            self.batch_index = len(self.testData)
        else:
            currentTestData  = self.preprocess(self.testData[self.batch_index:self.batch_index+self.batch_size])
        if self.batch_index==len(self.testData):
            self.runThrough = True
        self.batch_index      += self.batch_size
        return currentTestData

    def preprocess(self, pdata):
        """Perform generic preprocessing on the images

        # Arguments
            pdata: batch of data to be preprocessed

        # Returns
            Array containing the labels and the preprocessed data.
        """
        labels = []
        data   = []
        for i in pdata:
            d = pTasks(i, self.reshapeDims, self.normalize)
            labels.append(int(i[0]))
            data.append(d)
        return (np.array(labels), np.array(data))


class ODDataGenerator(object):
    """docstring for ODDataGenerator"""
    
    def __init__(self, testDir, reshapeDims, batch_size, classes, num_classes):
        super(ODDataGenerator, self).__init__()
        self.images      = testDir['images'] #each is a list
        self.imageSet    = testDir['testNames']
        self.annoDirs    = testDir['annotations']
        self.reshapeDims = reshapeDims
        self.batch_size  = batch_size
        self.classes     = classes
        self.num_classes = num_classes
        self.batch_index = 0
        self.runThrough  = False
        self.parseData()

    def parseData(self):
        self.filenames = []
        self.imageIds = []
        self.labels    = []
        self.eval_neutral = []

        for imdir, imfset, anndir in zip(self.images, self.imageSet, self.annoDirs):
            with open(imfset) as f:
                imagesIds = [line.strip() for line in f]
                self.imageIds+=imagesIds

            it = tqdm(imagesIds, desc="Processing image set '{}'".format(os.path.basename(imfset)), file=sys.stdout)

            for imageId in it:
                # temp = [item_dict['image_id']]
                filename = '{}'.format(imageId) + '.jpg'
                self.filenames.append(os.path.join(imdir, filename))

                with open(os.path.join(anndir, imageId+".xml")) as f:
                    soup = BeautifulSoup(f, 'xml')
                boxes = []
                eval_neutr = []
                objects = soup.find_all('object')

                for obj in objects:
                    class_name = obj.find('name', recursive=False).text
                    class_id = self.classes.index(class_name)
                    # pose = obj.find('pose', recursive=False).text
                    bndbox = obj.find('bndbox', recursive=False)
                    xmin = int(bndbox.xmin.text)
                    ymin = int(bndbox.ymin.text)
                    xmax = int(bndbox.xmax.text)
                    ymax = int(bndbox.ymax.text)
                    difficult = int(obj.find('difficult', recursive=False).text)

                    item_dict = {'image_name': filename,
                                 'image_id': imageId,
                                 'class_name': class_name,
                                 'class_id': class_id,
                                 'xmin': xmin,
                                 'ymin': ymin,
                                 'xmax': xmax,
                                 'ymax': ymax}
                    box = []
                    labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax')
                    for item in labels_output_format:
                        box.append(item_dict[item])
                    
                    # temp.append(box)
                    boxes.append(box)
                    if difficult: eval_neutr.append(True)
                    else: eval_neutr.append(False)
                self.labels.append(boxes)
                self.eval_neutral.append(eval_neutr)
        # print(len(self.filenames))

    def getNextBatch(self):
        currentTestData = []
        if self.batch_index>=len(self.filenames):
            self.batch_index = 0
            # currentTestData = [self.preprocess(i, l, imageId, self.reshapeDims) for i, l, imageId in 
            #                    zip(self.filenames[self.batch_index:self.batch_index+self.batch_size], 
            #                    self.labels[self.batch_index:self.batch_index+self.batch_size],
            #                    self.imageIds[self.batch_index:self.batch_index+self.batch_size])]
            currentTestData = self.preprocess(self.filenames[self.batch_index:self.batch_index+self.batch_size],
                                             self.labels[self.batch_index:self.batch_index+self.batch_size],
                                             self.imageIds[self.batch_index:self.batch_index+self.batch_size],
                                             self.reshapeDims)
        elif self.batch_index + self.batch_size >= len(self.filenames) and self.batch_index<len(self.filenames):
            # currentTestData = [self.preprocess(i, l, imageId, self.reshapeDims) for i, l, imageId in 
            #                    zip(self.filenames[self.batch_index:], self.labels[self.batch_index:], self.imageIds[self.batch_index:])]
            currentTestData = self.preprocess(self.filenames[self.batch_index:],
                                              self.labels[self.batch_index:],
                                              self.imageIds[self.batch_index:],
                                              self.reshapeDims)
            self.batch_index = len(self.filenames)
        else:
            # currentTestData = [self.preprocess(i, l, imageId, self.reshapeDims) for i, l, imageId in 
            #                    zip(self.filenames[self.batch_index:self.batch_index+self.batch_size], 
            #                    self.labels[self.batch_index:self.batch_index+self.batch_size],
            #                    self.imageIds[self.batch_index:self.batch_index+self.batch_size])]
            currentTestData = self.preprocess(self.filenames[self.batch_index:self.batch_index+self.batch_size],
                                             self.labels[self.batch_index:self.batch_index+self.batch_size],
                                             self.imageIds[self.batch_index:self.batch_index+self.batch_size],
                                             self.reshapeDims)
        if self.batch_index==len(self.filenames):
            self.runThrough = True
        else:
            self.batch_index    += self.batch_size

        return currentTestData

    def preprocess(self, testImage, labels, imageIds, reshapeDims):
        Ids = []
        labelList = []
        imgs = []

        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4

        for i in range(len(imageIds)):
            Ids.append(imageIds[i])

            # I = tf.keras.preprocessing.image.load_img(testImage[i])
            # I = I.resize(reshapeDims)
            # I = image.img_to_array(I)

            # imgs.append(I)

            I = cv2.imread(testImage[i])
            imgH, imgW = I.shape[:2]
            I = cv2.resize(I, dsize=tuple(reshapeDims), interpolation=cv2.INTER_LINEAR)
            imgs.append(I)          

            imageBoxes = np.copy(labels[i])
            imageBoxes[:, [ymin, ymax]] = np.round(imageBoxes[:, [ymin, ymax]] * (reshapeDims[0] / imgH), decimals=0)
            imageBoxes[:, [xmin, xmax]] = np.round(imageBoxes[:, [xmin, xmax]] * (reshapeDims[1] / imgW), decimals=0)
            labelList.append(imageBoxes)

        return ((np.array(Ids), np.array(labelList)), np.array(imgs))

       
