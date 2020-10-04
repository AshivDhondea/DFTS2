"""
run_experiment_01_transmission.py

Called by main_02_transmission.py

Simulate the transmission of packets through a channel.

1. Load a batch of tensor data out of the device model.
2. Quantize the data according to the chosen quantization level.
3. Packetize the data.
4. Transmit the data through the channel.
5. Save the transmitted packets.

Save repaired tensors at different TC iteration counts for classification later on.
Save completed tensors at iters = 50, 100, 150, 200

Date: Friday October 2, 2020.
"""

import sys
sys.path.append('..')

import time
import numpy as np
import os
import copy

from .utils import *
from .simmods import *

from .calloc import loadChannel, quantInit

from models.packetModel import PacketModel as PacketModel
from tensor_completion.tc_algos import *
# --------------------------------------------------------------------------- #

def runSimulation(loaded_model_name,experiment_params_dict, splitLayer, transDict, tensor_completion_dict, simDir):
    """
    Run a simulation experiment.

    Parameters
    ----------
    model : tf.keras model or path to a tf.keras model
        Tensorflow.keras model or path to it, along with its architecture and weights.
    epochs : integer
        Number of Monte Carlo runs for the MC experiment.
    splitLayer : string
        Name of layer at which the DNN model should be split.
    task : 0 or 1 (bool)
        Two types of task: classification (0) and object detection (1).
    modelDict : dictionary
        Dictionary of models provided natively by tf.keras
    transDict : dictionary
        Dictionary containing parameters for the transmission.
    simDir : path to a directory
        Path to an existing directory to save the simulation results.
    customObjects : TYPE
        DESCRIPTION.
    evaluator : object
        Evaluation allocator object.

    Returns
    -------
    None.

    """
    # parameters for the transmission.
    rowsPerPacket = transDict['rowsperpacket']
    quantization  = transDict['quantization']
    channel       = transDict['channel']

    MC_initial = experiment_params_dict['MC_ini']
    MC_final = experiment_params_dict['MC_fin']
    num_MC = MC_final - MC_initial + 1

    batch_initial = experiment_params_dict['batch_ini']
    batch_final = experiment_params_dict['batch_fin']
    save_damaged_tensor_flag = experiment_params_dict['save_damaged_tensor']

    silrtc_flag = tensor_completion_dict['SiLRTC']['include']
    halrtc_flag = tensor_completion_dict['HaLRTC']['include']

    if silrtc_flag == True:
        maxiters = tensor_completion_dict['SiLRTC']['maxiters']
        error_mc_silrtc = np.zeros([num_MC,maxiters],dtype=np.float64)
    if halrtc_flag == True:
        maxiters = tensor_completion_dict['HaLRTC']['maxiters']
        error_mc_halrtc = np.zeros([num_MC,maxiters],dtype=np.float64)

    if channel != 'noChannel':
        loss_probability = channel['GilbertChannel']['lossProbability']
        burst_length = channel['GilbertChannel']['burstLength']

        results_dir = os.path.join(simDir,loaded_model_name,splitLayer,'lp_'+str(loss_probability)+'_Bl_'+str(burst_length))
        os.makedirs(results_dir,exist_ok=True)
    else:
        results_dir = os.path.join(simDir,loaded_model_name,splitLayer)

    # Objects for the channel, quantization and error concealment.
    channel = loadChannel(channel)
    quant   = quantInit(quantization)

    # Run through all batches in the dataset.
    for i_b in range(batch_initial,batch_final+1):
        print(f"Now processing batch {i_b}")
        batch_start_time = time.time()
        # -------------------------------------------------------------------- #
        # Load the batch of tensor data.
        device_tensor = np.load(os.path.join(simDir,loaded_model_name,splitLayer,'device_tensor_batch_'+str(i_b)+'.npy'))
        # -------------------------------------------------------------------- #
        # If tensor completion is desired, reset the errors after each batch.
        if silrtc_flag == True:
            error_mc_silrtc = np.zeros([num_MC,maxiters],dtype=np.float64)
        if halrtc_flag == True:
            error_mc_halrtc = np.zeros([num_MC,maxiters],dtype=np.float64)

        for i_mc in range(MC_initial,MC_final+1):
            print(f"Monte Carlo run number {i_mc}")
            deviceOut = copy.deepcopy(device_tensor)

            print(np.shape(deviceOut))

            # Reset quantization parameters
            quanParams = []
            # --------------------------------------------------------------- #
            # On the mobile side:
            # quantize the output of the device model (if needed).
            ##
            # Quantize the data
            if quant!='noQuant':
                for i in range(len(deviceOut)):
                    quant.bitQuantizer(deviceOut[i])
                    deviceOut[i] = quant.quanData
                    quanParams.append([quant.min, quant.max])

            deviceOut_no_channel = copy.deepcopy(deviceOut)
            # ---------------------------------------------------------------- #
            # Transmit the tensor deviceOut through the channel.
            if channel!='noChannel':
                lossMatrix = []
                receivedIndices = []
                lostIndices = []
                dOut = []
                for i in range(len(deviceOut)):
                    print(i)
                    dO, lM, rI, lI = transmit(deviceOut[i], channel, rowsPerPacket)
                    dOut.append(dO)
                    lossMatrix.append(lM)
                    receivedIndices.append(rI)
                    lostIndices.append(lI)
                    channel.lossMatrix = []
                deviceOut = dOut

            for i in range(len(deviceOut)):
                damaged_tensor = deviceOut[i].packet_seq
                unpacketized_damaged_tensor = deviceOut[i].data_tensor
                # convert receivedIndices to a map
                received_map = np.zeros(np.shape(damaged_tensor),dtype=bool)
                receivedIndices_arr = np.asarray(receivedIndices)

                for i_rec in range(np.shape(receivedIndices_arr)[2]):
                    rec_ind = receivedIndices_arr[0,0,i_rec,:]
                    received_map[rec_ind[0],rec_ind[1],:,:,rec_ind[2]] = np.ones([np.shape(received_map)[2],np.shape(received_map)[3]],dtype=bool)

                map_PM = PacketModel(rows_per_packet = rowsPerPacket,data_shape = np.shape(unpacketized_damaged_tensor),packet_seq = received_map)
                unpacketized_map = map_PM.data_tensor

                 if save_damaged_tensor_flag ==  True:
                     # Save tensors to npy files
                     # Packets
                     np.save(os.path.join(results_dir,'packet_seq_'+str(i)+'_batch_'+str(i_b)+'_MC_'+str(i_mc)+'.npy'),damaged_tensor)
                     # receivedIndices
                     np.save(os.path.join(results_dir,'receivedIndices_'+str(i)+'_batch_'+str(i_b)+'_MC_'+str(i_mc)+'.npy'),receivedIndices)
                     # lostIndices
                     np.save(os.path.join(results_dir,'lostIndices'+str(i)+'_batch_'+str(i_b)+'_MC_'+str(i_mc)+'.npy'),lostIndices)
                     np.save(os.path.join(results_dir,'unpacketized_damaged_tensor_'+str(i)+'_batch_'+str(i_b)+'_MC_'+str(i_mc)+'.npy'),unpacketized_damaged_tensor)
                     np.save(os.path.join(results_dir,'unpacketized_map_'+str(i)+'_batch_'+str(i_b)+'_MC_'+str(i_mc)+'.npy'),unpacketized_map)

                if silrtc_flag == True or halrtc_flag == True:
                    repaired_tensor_silrtc_50 = np.zeros_like(unpacketized_damaged_tensor)
                    repaired_tensor_silrtc_100 = np.zeros_like(unpacketized_damaged_tensor)
                    repaired_tensor_silrtc_150 = np.zeros_like(unpacketized_damaged_tensor)
                    repaired_tensor_silrtc_200 = np.zeros_like(unpacketized_damaged_tensor)

                    repaired_tensor_halrtc_50 = np.zeros_like(unpacketized_damaged_tensor)
                    repaired_tensor_halrtc_100 = np.zeros_like(unpacketized_damaged_tensor)
                    repaired_tensor_halrtc_150 = np.zeros_like(unpacketized_damaged_tensor)
                    repaired_tensor_halrtc_200 = np.zeros_like(unpacketized_damaged_tensor)

                    for k in range(np.shape(unpacketized_map)[0]):
                        print(f"Tensor completion on item {k} of batch {i_b}")
                        item_start_time = time.time()

                        item_unpacketized_map = unpacketized_map[k,:,:,:]
                        subs = np.argwhere(item_unpacketized_map == True)
                        X_damaged = unpacketized_damaged_tensor[k,:,:,:]
                        vals = list(map(lambda x, y, z: X_damaged[x][y][z], subs[:,0], subs[:,1], subs[:,2]))

                        if silrtc_flag == True:
                            X_estimated_silrtc,error_iters_silrtc = fn_silrtc_damaged_error(X_damaged,maxiters,subs,vals)
                            repaired_tensor_silrtc_50[k,:,:,:] = X_estimated_silrtc[:,:,:,50-1]
                            repaired_tensor_silrtc_100[k,:,:,:] = X_estimated_silrtc[:,:,:,100-1]
                            repaired_tensor_silrtc_150[k,:,:,:] = X_estimated_silrtc[:,:,:,150-1]
                            repaired_tensor_silrtc_200[k,:,:,:] = X_estimated_silrtc[:,:,:,200-1]

                            error_mc_silrtc[i_mc,:] = error_mc_silrtc[i_mc,:] + error_iters_silrtc

                        if halrtc_flag == True:
                            X_estimated_halrtc,error_iters_halrtc = fn_halrtc_damaged_error(X_damaged,maxiters,subs,vals)
                            repaired_tensor_halrtc_50[k,:,:,:] = X_estimated_halrtc[:,:,:,50-1]
                            repaired_tensor_halrtc_100[k,:,:,:] = X_estimated_halrtc[:,:,:,100-1]
                            repaired_tensor_halrtc_150[k,:,:,:] = X_estimated_halrtc[:,:,:,150-1]
                            repaired_tensor_halrtc_200[k,:,:,:] = X_estimated_halrtc[:,:,:,200-1]

                            error_mc_halrtc[i_mc,:] = error_mc_halrtc[i_mc,:] + error_iters_halrtc

                        total_time = time.time() - item_start_time
                        print(f"For {maxiters} iterations, the time taken per item is {total_time}s")
             # --------------------------------------------------------------- #
            # On the cloud side:
            print("Received tensors at the cloud and saving them for processing later.")
            # if the tensor was quantized, inverse quantize it.

            if quant!='noQuant':
                for i in range(len(deviceOut)):
                    # No channel
                    quant.quanData = deviceOut_no_channel[i]
                    qMin, qMax = quanParams[i]
                    quant.min = qMin
                    quant.max = qMax
                    deviceOut_no_channel[i] = quant.inverseQuantizer()
                    np.save(os.path.join(results_dir,'cloud_tensor_no_channel'+str(i)+'_batch_'+str(i_b)+'_MC_'+str(i_mc)+'.npy'),deviceOut_no_channel[i])

                    # if a channel was used, inverse quantize the packets.
                    if channel!='noChannel':
                        # No tensor completion or error concealment.
                        quant.quanData = deviceOut[i].data_tensor
                        qMin, qMax = quanParams[i]
                        quant.min = qMin
                        quant.max = qMax
                        deviceOut[i].data_tensor = quant.inverseQuantizer()

                        np.save(os.path.join(results_dir,'cloud_tensor_'+str(i)+'_batch_'+str(i_b)+'_MC_'+str(i_mc)+'.npy'),deviceOut[i].data_tensor)
                        # ---------------------------------------------------- #
                        if silrtc_flag == True:
                            quant.quanData = repaired_tensor_silrtc_50
                            qMin, qMax = quanParams[i]
                            quant.min = qMin
                            quant.Max = qMax
                            repaired_tensor_silrtc_invQ_50 = quant.inverseQuantizer()

                            np.save(os.path.join(results_dir,'silrtc_cloud_tensor_'+str(i)+'_batch_'+str(i_b)+'_MC_'+str(i_mc)+'_iter_50.npy'),repaired_tensor_silrtc_invQ_50)
                            # ------------------------------------------------ #
                            quant.quanData = repaired_tensor_silrtc_100
                            qMin, qMax = quanParams[i]
                            quant.min = qMin
                            quant.Max = qMax
                            repaired_tensor_silrtc_invQ_100 = quant.inverseQuantizer()

                            np.save(os.path.join(results_dir,'silrtc_cloud_tensor_'+str(i)+'_batch_'+str(i_b)+'_MC_'+str(i_mc)+'_iter_100.npy'),repaired_tensor_silrtc_invQ_100)
                            # ------------------------------------------------ #
                            quant.quanData = repaired_tensor_silrtc_150
                            qMin, qMax = quanParams[i]
                            quant.min = qMin
                            quant.Max = qMax
                            repaired_tensor_silrtc_invQ_150 = quant.inverseQuantizer()

                            np.save(os.path.join(results_dir,'silrtc_cloud_tensor_'+str(i)+'_batch_'+str(i_b)+'_MC_'+str(i_mc)+'_iter_150.npy'),repaired_tensor_silrtc_invQ_150)
                            # ------------------------------------------------ #
                            quant.quanData = repaired_tensor_silrtc_200
                            qMin, qMax = quanParams[i]
                            quant.min = qMin
                            quant.Max = qMax
                            repaired_tensor_silrtc_invQ_200 = quant.inverseQuantizer()

                            np.save(os.path.join(results_dir,'silrtc_cloud_tensor_'+str(i)+'_batch_'+str(i_b)+'_MC_'+str(i_mc)+'_iter_200.npy'),repaired_tensor_silrtc_invQ_200)
                        # ---------------------------------------------------- #
                        if halrtc_flag == True:
                            quant.quanData = repaired_tensor_halrtc_50
                            qMin,qMax = quanParams[i]
                            quant.min = qMin
                            quant.max = qMax
                            repaired_tensor_halrtc_invQ_50 = quant.inverseQuantizer()

                            np.save(os.path.join(results_dir,'halrtc_cloud_tensor_'+str(i)+'_batch_'+str(i_b)+'_MC_'+str(i_mc)+'_iter_50.npy'),repaired_tensor_halrtc_invQ_50)
                            # ------------------------------------------------ #
                            quant.quanData = repaired_tensor_halrtc_100
                            qMin,qMax = quanParams[i]
                            quant.min = qMin
                            quant.max = qMax
                            repaired_tensor_halrtc_invQ_100 = quant.inverseQuantizer()

                            np.save(os.path.join(results_dir,'halrtc_cloud_tensor_'+str(i)+'_batch_'+str(i_b)+'_MC_'+str(i_mc)+'_iter_100.npy'),repaired_tensor_halrtc_invQ_100)
                            # ------------------------------------------------ #
                            quant.quanData = repaired_tensor_halrtc_150
                            qMin,qMax = quanParams[i]
                            quant.min = qMin
                            quant.max = qMax
                            repaired_tensor_halrtc_invQ_150 = quant.inverseQuantizer()

                            np.save(os.path.join(results_dir,'halrtc_cloud_tensor_'+str(i)+'_batch_'+str(i_b)+'_MC_'+str(i_mc)+'_iter_150.npy'),repaired_tensor_halrtc_invQ_150)
                            # ------------------------------------------------ #
                            quant.quanData = repaired_tensor_halrtc_200
                            qMin,qMax = quanParams[i]
                            quant.min = qMin
                            quant.max = qMax
                            repaired_tensor_halrtc_invQ_200 = quant.inverseQuantizer()

                            np.save(os.path.join(results_dir,'halrtc_cloud_tensor_'+str(i)+'_batch_'+str(i_b)+'_MC_'+str(i_mc)+'_iter_200.npy'),repaired_tensor_halrtc_invQ_200)
                            # ------------------------------------------------ #

                    else:
                        quant.quanData = deviceOut[i]
                        qMin, qMax = quanParams[i]
                        quant.min = qMin
                        quant.max = qMax
                        deviceOut[i] = quant.inverseQuantizer()

        np.save(os.path.join(results_dir,'error_mc_silrtc_'+str(i)+'_batch_'+str(i_b)+'_num_MC_'+str(num_MC)+'.npy'),error_mc_silrtc)
        np.save(os.path.join(results_dir,'error_mc_halrtc_'+str(i)+'_batch_'+str(i_b)+'_num_MC_'+str(num_MC)+'.npy'),error_mc_halrtc)
        # -------------------------------------------------------------------- #
        batch_end_time = time.time()
        print(f"Batch {i_b} took {batch_end_time - batch_start_time}s")
