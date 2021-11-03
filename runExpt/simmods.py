"""
Simulation module functions.

Documentation to be sorted out later.

Use timeit instead of time for timing purposes.
"""
import sys
sys.path.append('..')

import numpy as np
from timeit import default_timer as timer
import glob
import os
import tensorflow as tf
from models.packetModel import PacketModel as PacketModel
# ---------------------------------------------------------------------------- #
def transmit(compressOut, channel, rowsPerPacket):
    """
    Simulate packetization and transmission of the packets through a channel.

    # Arguments
        compressOut: List of mobile device tensors.
        channel: channel object
        rowsPerPacket: Integer number of rows of the feature map to be considered as one packet

    # Returns
        Packetized and lost data along with the indices of the lost and retained packets
    """
    start_time   = timer()
    pckts = PacketModel(rows_per_packet=rowsPerPacket,data_tensor=compressOut)

    lossMatrix = channel.simulate(pckts.packet_seq.shape[0]*pckts.packet_seq.shape[1]*pckts.packet_seq.shape[-1])
    lossMatrix = lossMatrix.reshape(pckts.packet_seq.shape[0], pckts.packet_seq.shape[1], pckts.packet_seq.shape[-1])

    receivedIndices = np.where(lossMatrix==True)
    receivedIndices = np.dstack((receivedIndices[0], receivedIndices[1], receivedIndices[2]))

    lostIndices = np.where(lossMatrix==False)
    lostIndices = np.dstack((lostIndices[0], lostIndices[1], lostIndices[2]))

    pckts.packet_seq[lostIndices[:,:,0], lostIndices[:,:,1], :, :, lostIndices[:,:,-1]] = 0

    total_time = timer() - start_time
    print(f"Transmission Complete in {total_time:.3f}s")
    return pckts, lossMatrix, receivedIndices, lostIndices

# ---------------------------------------------------------------------------- #
def fn_Data_PreProcessing_ImgClass(path_base,reshapeDims,normalize):
    # Load the dataset
    print('Available classes in the dataset are: ')
    classes_list = os.listdir(path_base)
    print(classes_list)

    # To load the images in the dataset
    dataset_x_files = []
    dataset_y_labels = []
    file_names = []
    # count how many examples there are for each class
    for i in range(len(classes_list)):
        examples = glob.glob1(os.path.join(path_base,classes_list[i]),"*."+"jpg")
        examples += glob.glob1(os.path.join(path_base,classes_list[i]),"*."+"jpeg")
        examples += glob.glob1(os.path.join(path_base,classes_list[i]),"*."+"png")
        examples += glob.glob1(os.path.join(path_base,classes_list[i]),"*."+"tif")
        #classes_count[i] = len(examples)

        for k in range(len(examples)):
            I = tf.keras.preprocessing.image.load_img(os.path.join(path_base,classes_list[i],examples[k]))
            I = I.resize(reshapeDims)
            im_array = tf.keras.preprocessing.image.img_to_array(I)

            if normalize == True:
                im_array /= 127.5
                im_array -= 1.

            dataset_x_files.append(im_array)
            dataset_y_labels.append(classes_list[i])
            file_names.append(examples[k])
    return dataset_x_files,dataset_y_labels,file_names
