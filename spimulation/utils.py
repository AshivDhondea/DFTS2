"""
Utilities functions for testConfig

Documentation to be sorted out later.

"""

#import os # imported but unused.
#import keras # hans commented out. 10 July 2020
import time
import numpy as np
#from keras.models import load_model

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import tensorflow as tf

def modelLoader(model, modelDict, customObjects):
    """Loads the desired keras model from the disk

    # Arguments
        model: desired keras model or path to that model
        modelDict: dictionary containing official keras models

    # Returns
        A keras model, containing its weights and architecture
    """
    if model in modelDict.values():
	# edited by Hans. 22 June 2020
        model = getattr(tf.keras.applications,f"{model}")()
        return model
    else:
	    #edited by Hans 10 July 2020
        model = tf.keras.models.load_model(model, custom_objects=customObjects)
        return model

def timing(func):
    t1  = time.time()
    res = func(*args, **kwargs)
    t2  = time.time()
    print(t1-t2)

def errorCalc(remoteOut, classValues):
    """
    Calculate the accuracy of the prediction.

    # Arguments
        remoteOut: prediction from the model in the cloud
        classValues: true labels of the data

    # Returns
        Accuracy of predictions, a number between 0 and 1
    """
    predictions = np.argmax(remoteOut, axis=1)
    return np.sum(np.equal(predictions, classValues))/classValues.shape[0]

def createFile(quant, conceal, splitLayer):
    """Creates a data file based on the desired options
    """
    fileName = splitLayer+"_"
    if quant!="noQuant":
        fileName += f"{quant.nBits}BitQuant_"
    if conceal!="noConceal":
        fileName += "EC"
    fileName += ".npy"
    return fileName
