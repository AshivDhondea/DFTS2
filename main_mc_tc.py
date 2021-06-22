"""
Similar to main_mc_caltec.py except that this uses either of silrtc
and halrtc.

"""
import argparse
import re
import yaml
import os,sys
import numpy as np
import pandas as pd
from models.packetModel import PacketModel as PacketModel
from models.BrokenModel import BrokenModel as BrokenModel
import glob
from timeit import default_timer as timer
#from PIL import Image
import tensorflow as tf
from spimulation.calloc import quantInit,loadChannel
from spimulation.simmods import *
from tensor_completion.tc_algos import *
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
                if j=='channel':# or j=='concealment': # or j=='TensorCompletion':
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
MC_params = paramsDict['Task']['MonteCarlo']
MC_start_index = MC_params['MC_start_index']
MC_end_index = MC_params['MC_end_index']
dataset    = paramsDict['TestInput']['dataset']
batch_size = paramsDict['TestInput']['batch_size']
testdir    = paramsDict['TestInput']['testdir']
splitLayer = paramsDict['SplitLayer']['split']
transDict  = paramsDict['Transmission']
simDir = paramsDict['OutputDir']['simDataDir']
results_dir = paramsDict['OutputDir']['resDataDir']
path_base = testdir['images']
tc_method = paramsDict['TensorCompletion']['method']
num_iters = paramsDict['TensorCompletion']['numiters']
rowsPerPacket = transDict['rowsperpacket']
quantization  = transDict['quantization']

channel       = transDict['channel']
lossProbability = channel['GilbertChannel']['lossProbability']
burstLength = channel['GilbertChannel']['burstLength']

# Objects for the channel, quantization and error concealment.
channel = loadChannel(channel)
quant_tensor1 = quantInit(quantization,tensor_id = 1)
quant_tensor2 = quantInit(quantization,tensor_id = 2)

numBits = quantization[1]['numberOfBits']
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
results_dir = os.path.join(results_dir,loaded_model_name,tc_method,splitLayer+'_lp_'+str(lossProbability)+'_Bl_'+str(burstLength))
os.makedirs(results_dir,exist_ok=True)

tf.keras.utils.plot_model(loaded_model,to_file=os.path.join(results_dir,splitLayer+'_full_model.png'),show_shapes=True)
with open(os.path.join(results_dir,splitLayer+'_full_model.txt'),'w') as fh:
    loaded_model.summary(print_fn = lambda x: fh.write(x + '\n'))
# ---------------------------------------------------------------------------- #
# Plot architecture of mobile and cloud sub models. Save their summary as txt file.
tf.keras.utils.plot_model(mobile_model,to_file=os.path.join(results_dir,splitLayer+'_mobile_model.png'),show_shapes=True)

with open(os.path.join(results_dir,splitLayer+'_mobile_model.txt'),'w') as fh:
    mobile_model.summary(print_fn = lambda x: fh.write(x + '\n'))

tf.keras.utils.plot_model(cloud_model,to_file=os.path.join(results_dir,splitLayer+'_cloud_model.png'),show_shapes=True)

with open(os.path.join(results_dir,splitLayer+'_cloud_model.txt'),'w') as fh:
    cloud_model.summary(print_fn = lambda x: fh.write(x + '\n'))

loss_maps_dir = os.path.join(simDir,path_base,loaded_model_name,splitLayer+'_lp_'+str(lossProbability)+'_Bl_'+str(burstLength))
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

mc_gc_accuracy = np.zeros([MC_end_index - MC_start_index],dtype=np.float64)
mc_full_accuracy = np.zeros([MC_end_index - MC_start_index],dtype=np.float64)
mc_tc_accuracy = np.zeros([MC_end_index - MC_start_index],dtype=np.float64)
mc_time = np.zeros([MC_end_index - MC_start_index],dtype=np.float64)

for i_mc in range(MC_start_index,MC_end_index):
    print(f"Now running Monte Carlo run {i_mc}")
    true_labels = []
    prediction_full_model = []
    prediction_split_gc = []
    prediction_tc = []
    full_model_confidence = []
    cloud_split_gc_confidence = []
    tc_confidence = []
    # -------------------------------------------------------------------------#
    for i_b in range(len(batched_y_labels)):
        print(f"Processing batch {i_b}")
        batch_labels = np.asarray(batched_y_labels[i_b],dtype=np.int64)
        true_labels.extend(batch_labels)
        batch_imgs = batched_x_files[i_b]
        batch_imgs_stacked = np.vstack([i[np.newaxis,...] for i in batch_imgs])
        print(f'True labels {batch_labels}')
        # -------------------------------------------------------------------- #
        full_model_out = loaded_model.predict(batch_imgs_stacked)
        batch_predictions = np.argmax(full_model_out,axis=1)
        batch_confidence = np.max(full_model_out,axis=1)
        prediction_full_model.extend(batch_predictions)
        full_model_confidence.extend(batch_confidence)
        # -------------------------------------------------------------------- #
        deviceOut = mobile_model.predict(batch_imgs_stacked)
        # -------------------------------------------------------------------- #
        devOut = []
        if not isinstance(deviceOut, list):
            devOut.append(deviceOut)
            deviceOut = devOut
            # deviceOut is the output tensor for a batch of dat

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
        # ---------------------------------------------------------------------#
        # packetize quantized tensor.
        pkt_obj_list = []
        for i in range(len(deviceOut)):
            pkt_obj_list.append(PacketModel(rows_per_packet=rowsPerPacket,data_tensor=deviceOut[i]))

        # -------------------------------------------------------------------- #
        # Load loss matrices from file.
        loss_matrix = []
        for i in range(len(deviceOut)):
            loss_matrix.append(np.load(os.path.join(loss_maps_dir,'MC_'+str(i_mc)+'_batch_'+str(i_b)+'_tensor_'+str(i)+'_lossMatrix.npy')))
        # -------------------------------------------------------------------- #
        # apply this loss matrix to the tensor.
        for i in range(len(pkt_obj_list)):
            loss_map = loss_matrix[i]
            #print(np.shape(loss_map))
            channel_width = np.shape(pkt_obj_list[i].packet_seq)[3]

            # loop through items in batch.
            for item_index in range(np.shape(loss_map)[0]):
                item_lost_map = loss_map[item_index,:,:]

                lost_pkt_indices,lost_channel_indices = np.where(item_lost_map == False)

                if len(lost_pkt_indices) != 0:
                    # drop packet in tensor.
                    for k in range(len(lost_pkt_indices)):
                        #print(f'batch {i_b} item {item_index} {lost_pkt_indices[k]}')
                        pkt_obj_list[i].packet_seq[item_index,lost_pkt_indices[k],:,:,lost_channel_indices[k]] = np.zeros([rowsPerPacket,channel_width])
        # -------------------------------------------------------------------- #
        # Inverse quantize received packets.
        cOut = []
        # If necessary, inverse quantize tensors.
        if len(pkt_obj_list) > 1:
            if quant_tensor1!= 'noQuant':
                print("Inverse quantizing tensors")
                if channel != 'noChannel':
                    quant_tensor1.quanData = pkt_obj_list[0].data_tensor
                    qMin, qMax = quanParams_1
                    quant_tensor1.min = qMin
                    quant_tensor1.max = qMax
                    cOut.append(quant_tensor1.inverseQuantizer())

                    quant_tensor2.quanData = pkt_obj_list[1].data_tensor
                    qMin, qMax = quanParams_2
                    quant_tensor2.min = qMin
                    quant_tensor2.max = qMax
                    cOut.append(quant_tensor2.inverseQuantizer())
                # else:# no need to handle this, because there will always be a channel for this experiment.
        else:
            if quant_tensor1 != 'noQuant':
                print("Inverse quantizing tensor")
                if channel != 'noChannel':
                    quant_tensor1.quanData = pkt_obj_list[0].data_tensor
                    qMin, qMax = quanParams_1
                    quant_tensor1.min = qMin
                    quant_tensor1.max = qMax
                    cOut.append(quant_tensor1.inverseQuantizer())
        data_tensor_invQuant = cOut

        tensor_out = cloud_model.predict(data_tensor_invQuant)
        cloud_notc = np.argmax(tensor_out,axis=1)
        cloud_notc_confidence = np.max(tensor_out,axis=1)
        print(f"no tensor completion. pred {cloud_notc}")
        print(f"no tensor completion. pred confidence {cloud_notc_confidence}")
        prediction_split_gc.extend(cloud_notc)
        # -------------------------------------------------------------------- #
        mc_start_time = timer()
        # Now run tensor completion method.
        tc_packet_models = []

        for i in range(len(cOut)):
            tc_packet_models.append(PacketModel(rows_per_packet=rowsPerPacket,data_tensor=np.copy(data_tensor_invQuant[i])))

        num_channels = np.shape(tc_packet_models[0].packet_seq)[-1]
        num_pkts = np.shape(tc_packet_models[0].packet_seq)[1]
        num_examples = np.shape(tc_packet_models[0].packet_seq)[0]

        for i in range(len(cOut)):
            loss_map = loss_matrix[i]
            # loop through items in batch.
            for item_index in range(np.shape(loss_map)[0]):
                item_lost_map = loss_map[item_index,:,:]
                lost_pkt_indices,lost_channel_indices = np.where(item_lost_map == False)
                if len(lost_pkt_indices) != 0:
                    # drop packet in tensor.
                    for k in range(len(lost_pkt_indices)):
                        tc_packet_models[i].packet_seq[item_index,lost_pkt_indices[k],:,:,lost_channel_indices[k]] = np.zeros([rowsPerPacket,channel_width],dtype=np.float32)

            ##repaired_tensor = fn_complete_tensor_altec_pkts_star(np.copy(tc_packet_models[0].data_tensor),rowsPerPacket,altec_pkt_w,loss_map)
            ##tc_packet_models[i].data_tensor = repaired_tensor

        for i in range(len(tc_packet_models)):
            channel_width = np.shape(tc_packet_models[i].data_tensor)[2]
            loss_map = loss_matrix[i]

            for item_index in range(len(batch_labels)):
                # loop through items in batch.
                X_damaged = tc_packet_models[i].data_tensor[item_index,:,:,:]
                lost_map = loss_map[item_index,:,:]
                received_map = np.zeros(np.shape(tc_packet_models[i].data_tensor[item_index,:,:,:]),dtype=bool)
                for pkt_index in range(np.shape(lost_map)[0]):
                    for channel_index in range(np.shape(lost_map)[1]):
                        if lost_map[pkt_index,channel_index] == True:
                            received_map[pkt_index*rowsPerPacket:(pkt_index+1)*rowsPerPacket,:,channel_index] = np.ones([rowsPerPacket,channel_width],dtype=bool)

                subs = np.argwhere(received_map == True)
                vals = list(map(lambda x, y, z: X_damaged[x][y][z], subs[:,0], subs[:,1], subs[:,2]))

                if tc_method == 'SiLRTC':
                    X_estimated_tc, error_iters_tc = fn_silrtc_damaged_error(X_damaged,num_iters,subs,vals)
                    tc_packet_models[i].data_tensor[item_index,:,:,:] = X_estimated_tc[:,:,:,-1]

                if tc_method == 'HaLRTC':
                    X_estimated_tc, error_iters_tc = fn_halrtc_damaged_error(X_damaged,num_iters,subs,vals)
                    tc_packet_models[i].data_tensor[item_index,:,:,:] = X_estimated_tc[:,:,:,-1]

        repaired_tensor_list = []
        for i in range(len(cOut)):
            repaired_tensor_list.append(tc_packet_models[i].data_tensor)

        cloud_model_out = cloud_model.predict(repaired_tensor_list)
        predictions_tc = np.argmax(cloud_model_out,axis=1)
        cloud_tc_confidence = np.max(cloud_model_out,axis=1)
        print(f"with {tc_method}, predictions_cloud_model are {predictions_tc}")
        print(f"{tc_method} repaired. pred confidence {cloud_tc_confidence}")
        prediction_tc.extend(predictions_tc)

        mc_end_time = timer()
        mc_time[i_mc - MC_start_index] += (mc_end_time - mc_start_time)

    mc_gc_accuracy[i_mc - MC_start_index] = np.sum(np.equal(prediction_split_gc,true_labels))/len(true_labels)
    mc_full_accuracy[i_mc - MC_start_index] = np.sum(np.equal(prediction_full_model,true_labels))/len(true_labels)
    mc_tc_accuracy[i_mc -MC_start_index] = np.sum(np.equal(prediction_tc,true_labels))/len(true_labels)


print(f"Summary of Monte Carlo experiment on {loaded_model_name} tensors at the split layer {splitLayer}.")
print(f"Prediction accuracy with full model: {mc_full_accuracy}")
print(f"Gilbert Channel cloud prediction accuracy: {mc_gc_accuracy}")
print(f"With {num_iters} of {tc_method} prediction accuracy: {mc_tc_accuracy}")
print(f"{num_iters} of {tc_method} took {mc_time}s")

mc_time_example = (np.mean(mc_time))/classes_count_total
print(f'{tc_method} processed one example in {mc_time_example}s.')

df = pd.DataFrame({'mc_full_accuracy':mc_full_accuracy,'mc_gc_accuracy':mc_gc_accuracy,'mc_'+tc_method+'_accuracy':mc_tc_accuracy})
df.to_csv(os.path.join(results_dir,tc_method+'_mc_'+str(MC_start_index)+'_'+str(MC_end_index)+'.csv'),index=False)

np.save(os.path.join(results_dir,tc_method+'_time_taken_'+str(MC_start_index)+'_'+str(MC_end_index)+'.npy'),mc_time)
