"""
main_train_altec.py

For a chosen deep model and a split layer, this script splits the deep model at
the chosen split layer and does the mobile device model computations. The
resulting deep feature tensor is fed to the training algorithm.

This is the original version of ALTeC training method, with modifications to
account for the packetization scheme adopted in DFTS.

ALTeC weights are calculated in two ways:
1. the original ALTeC way - tensors are not quantized before being passed to the
ALTeC weight training function.
2. ALTeC + quantization way - tensors are quantized and inverse quantized before
being passed to the ALTeC weight training function. The idea is to replicate the
effect of quantization and then inverse quantization with deep feature tensors.

"""
from timeit import default_timer as timer
import os, sys
import numpy as np
import tensorflow as tf
import argparse
import re
import yaml
from models.BrokenModel import BrokenModel as BrokenModel
import glob
from spimulation.calloc import quantInit
import random
import pandas as pd
from tensor_completion.altec import *
# ---------------------------------------------------------------------------- #
# Process yml file which gives the parameters for this experiment.
def selectParamConfig(p, paramDict):
    """
    Throw everything except for the user selected parameter.

    Parameters
    ----------
    p: parameter for which selection is being made
    paramDict: dictionary containing user selected parameters

    Returns
    -------
    Selected parameter and the corresponding values

    Raises
    -------
    ParserError: if more than one option is selected for a parameter
    """
    sum = 0
    index = 0

    for i in paramDict:
        sum += paramDict[i]['include']
        if paramDict[i]['include']:
            index = i
    try:
        if sum>1:
            raise ParserError("Multiple configurations selected for {}".format(p), sum)
    except Exception as e:
        raise
    else:
        if sum==0:
            return (index, False)
        else:
            return (index, paramDict[index])

def configSettings(config):
    """
    Refine the parameter dictionary to only include user selected parameters.

    Parameters
    ----------
    config: dictionary read from the YAML file

    Returns
    ----------
    Dictionary containing only the options selected by the user
    """
    for i in config:
        if i=='Transmission':
            for j in config[i]:
                transDict = {}
                if j=='channel':
                    index, temp = selectParamConfig(j, config[i][j])
                    transDict[index] = temp
                    config[i][j] = transDict
    return config

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--params", help="path to the config file containing parameters", required=True)
args = vars(ap.parse_args())

filename = args['params']

with open(filename) as c:
    config = yaml.load(c,yaml.SafeLoader)

paramsDict = configSettings(config)

modelPath = paramsDict['Model']['kerasmodel']
customObjects = paramsDict['Model']['customObjects']
dataset    = paramsDict['TrainInput']['dataset']
batch_size = paramsDict['TrainInput']['batch_size']
path_base    = paramsDict['TrainInput']['traindir']
num_training_examples = paramsDict['TrainInput']['numtrain']
splitLayer = paramsDict['SplitLayer']['split']
transDict  = paramsDict['Transmission']
simDir = paramsDict['OutputDir']['trainweights']

rowsPerPacket = transDict['rowsperpacket']
quantization  = transDict['quantization']

quant_res_tensor_0 = transDict['quantization'][1]['numberOfBits']
quant_res_tensor_1 = transDict['quantization'][2]['numberOfBits']

# Objects for the channel, quantization and error concealment.
quant_tensor1 = quantInit(quantization,tensor_id = 1)
quant_tensor2 = quantInit(quantization,tensor_id = 2)
# ---------------------------------------------------------------------------- #
model_path = os.path.join(modelPath)
loaded_model = tf.keras.models.load_model(model_path)
# Object for splitting a tf.keras model into a mobile sub-model and a cloud
# sub-model at the chosen split layer 'splitLayer'.
testModel = BrokenModel(loaded_model, splitLayer, customObjects)
testModel.splitModel()

mobile_model = testModel.deviceModel
cloud_model = testModel.remoteModel

loaded_model_config = loaded_model.get_config()
loaded_model_name = loaded_model_config['name']

# Create results directory
results_dir = os.path.join(simDir,loaded_model_name,splitLayer)
os.makedirs(results_dir,exist_ok=True)

tf.keras.utils.plot_model(loaded_model,to_file=os.path.join(results_dir,splitLayer+'_full_model.png'),show_shapes=True)
with open(os.path.join(results_dir,splitLayer+'_full_model.txt'),'w') as fh:
    loaded_model.summary(print_fn = lambda x: fh.write(x + '\n'))
# ------------------------------------------------------------------------ #
# Plot architecture of mobile and cloud sub models. Save their summary as txt file.
tf.keras.utils.plot_model(mobile_model,to_file=os.path.join(results_dir,splitLayer+'_mobile_model.png'),show_shapes=True)

with open(os.path.join(results_dir,splitLayer+'_mobile_model.txt'),'w') as fh:
    mobile_model.summary(print_fn = lambda x: fh.write(x + '\n'))

tf.keras.utils.plot_model(cloud_model,to_file=os.path.join(results_dir,splitLayer+'_cloud_model.png'),show_shapes=True)

with open(os.path.join(results_dir,splitLayer+'_cloud_model.txt'),'w') as fh:
    cloud_model.summary(print_fn = lambda x: fh.write(x + '\n'))
# ---------------------------------------------------------------------------- #
# Process the training set.
width = 224
height = 224
reshapeDims = (width,height)

# List all images of any file type in the dataset.
training_examples = glob.glob1(path_base,"*."+"jpg")
training_examples += glob.glob1(path_base,"*."+"jpeg")
training_examples += glob.glob1(path_base,"*."+"JPEG")
training_examples += glob.glob1(path_base,"*."+"png")
training_examples += glob.glob1(path_base,"*."+"tif")

dataset_size = len(training_examples)
print(f'There are {dataset_size} examples in the dataset.')

# Use a random seed to randomly select half of the data set for training.
random.seed(11)
selected_list = random.sample(training_examples,num_training_examples)
print(f'length of training list {len(selected_list)}')

test_list = list(set(training_examples)-set(selected_list))

# Save the name of images used in training ALTeC weights. (therefore, these
# images cannot be used for evaluation later on.)
df = pd.DataFrame(selected_list)
df.to_csv(os.path.join(simDir,loaded_model_name,splitLayer,'training_set_images.csv'))
# Save the name of images not used in training. These images therefore can be
# used for evaluation later on.
df = pd.DataFrame(test_list)
df.to_csv(os.path.join(simDir,loaded_model_name,splitLayer,'test_set_images.csv'))

dataset_x_files = []
for k in range(len(selected_list)):
    I = tf.keras.preprocessing.image.load_img(os.path.join(path_base,selected_list[k]))
    I = I.resize(reshapeDims)
    im_array = tf.keras.preprocessing.image.img_to_array(I)

    if loaded_model_name in ['vgg16','densenet121']:
        #im_array = np.array(test_im)
        im_array /= 127.5
        im_array -= 1.
    dataset_x_files.append(im_array)
# ---------------------------------------------------------------------------- #
# using list comprehension
batched_x_files = [dataset_x_files[i: i + batch_size] for i in range(0,len(dataset_x_files),batch_size)]
num_batches = len(batched_x_files)
print(f'There are {num_batches} batches in the training set.')

tensor_0_weights = []
new_altec_time_taken = 0

for i_b in range(num_batches):
    print(f'Batch {i_b}')
    batch_imgs = batched_x_files[i_b]
    batch_imgs_stacked = np.vstack([i[np.newaxis,...] for i in batch_imgs])
    # ------------------------------------------------------------------------ #
    deviceOut = mobile_model.predict(batch_imgs_stacked)
    # ------------------------------------------------------------------------ #
    # Quantize the data
    quanParams_1 = []
    quanParams_2 = []
    print("Quantizing tensor.")
    quant_tensor1.bitQuantizer(deviceOut)
    deviceOut = quant_tensor1.quanData
    quanParams_1.append(quant_tensor1.min)
    quanParams_1.append(quant_tensor1.max)

    print("Inverse quantizing tensor.")
    quant_tensor1.quanData = deviceOut
    qMin, qMax = quanParams_1
    quant_tensor1.min = qMin
    quant_tensor1.max = qMax
    deviceOut = quant_tensor1.inverseQuantizer()
    # ------------------------------------------------------------------------ #
    # Compute ALTeC weights for the no quantization case (original ALTeC).
    print('ALTeC star weight training')
    tensor_in_model = PacketModel(rows_per_packet=rowsPerPacket,data_tensor=np.copy(deviceOut))
    new_altec_start = timer()
    batch_altec_weights = fn_compute_altec_weights_pkts(tensor_in_model)
    new_altec_time_taken += timer() - new_altec_start

    tensor_0_weights.extend([batch_altec_weights for _ in range(len(batch_imgs))])
    # ------------------------------------------------------------------------ #

print(f'There are {len(tensor_0_weights)} items in the training set.')
new_altec_weights = np.mean(tensor_0_weights,axis=0)
print(np.shape(new_altec_weights))

print(f'ALTeC took {new_altec_time_taken:.3f}s')

# Save weights for tensor
np.save(os.path.join(results_dir,splitLayer+'_rpp_'+str(rowsPerPacket)+'_'+str(quant_res_tensor_0)+'Bits_tensor_weights.npy'),new_altec_weights)

# ---------------------------------------------------------------------------- #
