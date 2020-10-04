"""
Sun Aug 16 19:18

Processing tensors being transmitted from the mobile model to the cloud model.

"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
# --------------------------------------------------------------- #
def tensor_to_tiled(tensor_np,num_channel_rows,num_channel_columns,height_channel,width_channel):
    """ Function for changing tensor to tiled image
        :param A,B dummy variable
    """
    A = np.reshape(tensor_np,(num_channel_rows,num_channel_columns,height_channel,width_channel))
    B = np.swapaxes(A,1,2)
    tiled = np.reshape(B,(num_channel_rows*height_channel, num_channel_columns*width_channel))
    return tiled

def tiled_to_tensor(tiled_image,num_channel_rows,num_channel_columns,height_channel,width_channel):
    """ Function for changing tiled image to tensor
        :param A,B dummy variable
    """
    A = np.reshape(tiled_image,(num_channel_rows,height_channel,num_channel_columns,width_channel))
    B = np.swapaxes(A,1,2)
    tensor = np.reshape(B,(num_channel_rows*num_channel_columns, height_channel, width_channel))
    return tensor

def fnTensorTiling(tensor_in,cr,cc,data_index,batch_index,splitLayer,tensor_name,JPG_quality):
    # tile, min/max, normalize, 8-bit
    tiled_tensor = []
    tensor_min = []
    tensor_max = []
    w,h,d = tensor_in[0].shape  # layer 3: 56 x 56 x 256
    X_all = [];
    for i in range(0,data_index):#deviceOut.shape[0]):
        # moveaxis
        temp = tensor_in[i,:,:,:] # (56,56,256)
        temp2 = np.moveaxis(temp,-1,0) # becomes (256,56,56)
        #print('temp2')
        #print(np.shape(temp2))

        X = tensor_to_tiled(temp2,cr,cc, w,h) # becomes (cr*56,cc*56)
        X_all.append(X)
        # don't need min with ReLU activation - assume 0
        tensor_min.append(np.amin(X))  # always 0 due to BatchNorm + ReLU
        #print('np.amin(X)')
        #print(np.amin(X))
        tensor_max.append(np.amax(X))
        # normalize to [0, 255]
        Xs = (((X - np.amin(X)) * 255)/(np.amax(X) - np.amin(X)))
        #Xs = (X * 255.0)/np.amax(X)
        # round to nearest integer
        np.around(Xs, out=Xs)
        # cast as 8-bit int
        Xs = Xs.astype('uint8')
        tiled_tensor.append(Xs)

    tiled_tensor = np.asarray(tiled_tensor)
    tiled_tensor = np.expand_dims(tiled_tensor, axis=3)

    # write test images
    # make path if it does not exist
    os.makedirs(os.path.join('simData',splitLayer), exist_ok=True)

    for i in range(0,data_index):#y_test.shape[0]):
        filename = tensor_name+'_'+str(batch_index)+'_'+str(i)+'.jpg'
        tf.keras.preprocessing.image.save_img(os.path.join('simData',splitLayer,filename), tiled_tensor[i], scale=False,file_formt='jpeg', quality=JPG_quality)
    return np.asarray(X_all)

def fnTensorProcessing(tensor_data,cr,cc,data_index,batch_index,splitLayer,tensor_str,JPG_quality):
    tensor_tiled = fnTensorTiling(tensor_data,cr,cc,data_index,batch_index,splitLayer,tensor_str,JPG_quality)

    for i in range(0,data_index):
        df = pd.DataFrame(tensor_tiled[i,:,:])
        filename = tensor_str+'_'+str(batch_index)+'_img_'+str(i)+'_tiled.csv'
        df.to_csv(os.path.join('simData',splitLayer,filename),index=True)
