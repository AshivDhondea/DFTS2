"""
Create loss maps according to parameters.
These loss maps can be used to benchmark the performance of tensor completion
methods.

"""
import argparse
import re
import yaml
import os,sys
import numpy as np
import pandas as pd
from models.BrokenModel import BrokenModel as BrokenModel
import glob
import tensorflow as tf
from spimulation.calloc import quantInit,loadChannel
from spimulation.simmods import *
# ---------------------------------------------------------------------------- #
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
# ---------------------------------------------------------------------------- #
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--params", help="path to the config file containing parameters", required=True)
args = vars(ap.parse_args())

filename = args['params']

with open(filename) as c:
    config = yaml.load(c,yaml.SafeLoader)

paramsDict = configSettings(config)

modelPath = paramsDict['Model']['kerasmodel']
customObjects = paramsDict['Model']['customObjects']
num_MC = paramsDict['Task']['MonteCarloRuns']
dataset    = paramsDict['TestInput']['dataset']
batch_size = paramsDict['TestInput']['batch_size']
testdir    = paramsDict['TestInput']['testdir']
splitLayer = paramsDict['SplitLayer']['split']
transDict  = paramsDict['Transmission']
simDir = paramsDict['OutputDir']['simDataDir']

path_base = testdir['images']

rowsPerPacket = transDict['rowsperpacket']
quantization  = transDict['quantization']

channel       = transDict['channel']
lossProbability = channel['GilbertChannel']['lossProbability']
burstLength = channel['GilbertChannel']['burstLength']

# Objects for the channel, quantization and error concealment.
channel = loadChannel(channel)
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

# ---------------------------------------------------------------------------- #
# Create results directory
results_dir = os.path.join(simDir,path_base,loaded_model_name,splitLayer+'_lp_'+str(lossProbability)+'_Bl_'+str(burstLength))
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
# Load the dataset
print('Available classes in the dataset are: ')
classes_list = os.listdir(path_base)
print(classes_list)

classes_count = np.zeros([len(classes_list)],dtype=int)

width = 224
height = 224
reshapeDims = (width,height)

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
    classes_count[i] = len(examples)

    for k in range(len(examples)):
        I = tf.keras.preprocessing.image.load_img(os.path.join(path_base,classes_list[i],examples[k]))
        I = I.resize(reshapeDims)
        im_array = tf.keras.preprocessing.image.img_to_array(I)

        if loaded_model_name in ['vgg16','densenet121']:
            #im_array = np.array(test_im)
            im_array /= 127.5
            im_array -= 1.

        dataset_x_files.append(im_array)
        dataset_y_labels.append(classes_list[i])
        file_names.append(examples[k])

classes_count_total = np.sum(classes_count)
print(f'The dataset comprises of {classes_count_total} images.')
# ---------------------------------------------------------------------------- #
# using list comprehension
batched_y_labels = [dataset_y_labels[i:i + batch_size] for i in range(0, len(dataset_y_labels), batch_size)]
batched_x_files = [dataset_x_files[i: i + batch_size] for i in range(0,len(dataset_x_files),batch_size)]

mc_gc_accuracy = np.zeros([num_MC],dtype=np.float64)
mc_full_accuracy = np.zeros([num_MC],dtype=np.float64)
mc_frobenius = np.zeros([num_MC])

for i_mc in range(num_MC):
    true_labels = []
    prediction_full_model = []
    prediction_split_gc = []
    full_model_confidence = []
    cloud_split_gc_confidence = []

    for i_b in range(len(batched_y_labels)):
        # Run through Monte Carlo runs.
        print(f"Monte Carlo run {i_mc} on batch {i_b}")
        batch_labels = np.asarray(batched_y_labels[i_b],dtype=np.int64)
        true_labels.extend(batch_labels)
        batch_imgs = batched_x_files[i_b]
        batch_imgs_stacked = np.vstack([i[np.newaxis,...] for i in batch_imgs])
        # ------------------------------------------------------------------------ #
        full_model_out = loaded_model.predict(batch_imgs_stacked)
        batch_predictions = np.argmax(full_model_out,axis=1)
        batch_confidence = np.max(full_model_out,axis=1)
        prediction_full_model.extend(batch_predictions)
        # full_model_confidence.extend(batch_confidence)
         # ------------------------------------------------------------------------ #
        deviceOut = mobile_model.predict(batch_imgs_stacked)
        # ------------------------------------------------------------------------ #
        devOut = []
        if not isinstance(deviceOut, list):
            devOut.append(deviceOut)
            deviceOut = devOut
            # deviceOut is the output tensor for a batch of dat

        # Accumulate quantized original tensors to calculate the Frobenius norm later.
        original_tensor_invQuant = []
        for i in range(len(deviceOut)):
            original_tensor_invQuant.append(deviceOut[i])

       # Quantize the data
        quanParams_1 = []
        quanParams_2 = []
        # If quantization is required:
        if len(deviceOut) > 1:
            if quant_tensor1!= 'noQuant':
                print("Quantizing tensors")
                quant_tensor1.bitQuantizer(deviceOut[0])
                deviceOut[0] = quant_tensor1.quanData
                quanParams_1.append(quant_tensor1.min)
                quanParams_1.append(quant_tensor1.max)

                quant_tensor2.bitQuantizer(deviceOut[1])
                deviceOut[1] = quant_tensor2.quanData
                quanParams_2.append(quant_tensor2.min)
                quanParams_2.append(quant_tensor2.max)
        else:
            if quant_tensor1!= 'noQuant':
                print("Quantizing tensor.")
                quant_tensor1.bitQuantizer(deviceOut[0])
                deviceOut[0] = quant_tensor1.quanData
                quanParams_1.append(quant_tensor1.min)
                quanParams_1.append(quant_tensor1.max)

        # ------------------------------------------------------------------------ #
        # Transmit the tensor deviceOut through the channel.
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
            # Save loss map.
            np.save(os.path.join(results_dir,'MC_'+str(i_mc)+'_batch_'+str(i_b)+'_tensor_'+str(i)+'_lossMatrix.npy'),lM)
        deviceOut_quant = dOut
        # -------------------------------------------------------------------- #
        # Inverse quantize received packets.
        # If necessary, inverse quantize tensors.
        if len(dOut) > 1:
            if quant_tensor1!= 'noQuant':
                print("Inverse quantizing tensors")
                if channel != 'noChannel':
                    quant_tensor1.quanData = deviceOut_quant[0].data_tensor
                    qMin, qMax = quanParams_1
                    quant_tensor1.min = qMin
                    quant_tensor1.max = qMax
                    deviceOut_quant[0].data_tensor = quant_tensor1.inverseQuantizer()

                    quant_tensor2.quanData = deviceOut_quant[1].data_tensor
                    qMin, qMax = quanParams_2
                    quant_tensor2.min = qMin
                    quant_tensor2.max = qMax
                    deviceOut_quant[1].data_tensor = quant_tensor2.inverseQuantizer()
                else:
                    # no channel.
                    quant_tensor1.quanData = deviceOut_quant[0]
                    qMin, qMax = quanParams_1
                    quant_tensor1.min = qMin
                    quant_tensor1.max = qMax
                    deviceOut_quant[0] = quant_tensor1.inverseQuantizer()

                    quant_tensor2.quanData = deviceOut_quant[1]
                    qMin, qMax = quanParams_2
                    quant_tensor2.min = qMin
                    quant_tensor2.max = qMax
                    deviceOut_quant[1] = quant_tensor2.inverseQuantizer()

        else:
            if quant_tensor1 != 'noQuant':
                print("Inverse quantizing tensor")
                if channel != 'noChannel':
                    quant_tensor1.quanData = deviceOut_quant[0].data_tensor
                    qMin, qMax = quanParams_1
                    quant_tensor1.min = qMin
                    quant_tensor1.max = qMax
                    deviceOut_quant[0].data_tensor = quant_tensor1.inverseQuantizer()
                # else:
                    # # no channel.
                    # quant_tensor1.quanData = deviceOut_quant[0]
                    # qMin, qMax = quanParams_1
                    # quant_tensor1.min = qMin
                    # quant_tensor1.max = qMax
                    # deviceOut_quant[0] = quant_tensor1.inverseQuantizer()

        cOut = []
        for i in range(len(dOut)):
            if channel != 'noChannel':
                cOut.append(np.copy(deviceOut_quant[i].data_tensor))
            # else:
            #     cOut.append(np.copy(deviceOut_quant[i]))

        deviceOut_invQuant = cOut
        # -------------------------------------------------------------------- #
        # Run cloud prediction on channel output data.
        tensor_out = cloud_model.predict(deviceOut_invQuant)
        cloud_out = np.argmax(tensor_out,axis=1)
        cloud_out_confidence = np.max(tensor_out,axis=1)

        prediction_split_gc.extend(cloud_out)

        for i in range(len(deviceOut_invQuant)):
            sse = np.sum(np.square(np.subtract(original_tensor_invQuant[i], deviceOut_invQuant[i])))
            mc_frobenius[i_mc] += sse

    mc_gc_accuracy[i_mc] = np.sum(np.equal(prediction_split_gc,true_labels))/len(true_labels)
    mc_full_accuracy[i_mc] = np.sum(np.equal(prediction_full_model,true_labels))/len(true_labels)

df = pd.DataFrame({'mc_full_accuracy':mc_full_accuracy,'mc_gc_accuracy':mc_gc_accuracy})
df.to_csv(os.path.join(results_dir,'gc_mc.csv'),index=False)
mc_frobenius = np.sqrt(mc_frobenius)/classes_count_total

np.save(os.path.join(results_dir,'gc_frobenius_mc.npy'),mc_frobenius)

print(f"Summary of Monte Carlo experiment on {loaded_model_name} tensors at the split layer {splitLayer}.")
print(f"Prediction accuracy with full model: {mc_full_accuracy}")
print(f"Gilbert Channel cloud prediction accuracy: {mc_gc_accuracy}")
print(f"Frobenius norm {mc_frobenius}")
# ---------------------------------------------------------------------------- #
