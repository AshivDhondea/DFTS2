"""Run Demonstration Image Classification Experiments.
"""
import sys,os
sys.path.append('..')

import numpy as np
from models.BrokenModel import BrokenModel as BrokenModel
import glob
import tensorflow as tf
import pandas as pd
from timeit import default_timer as timer
from .calloc import loadChannel,quantInit
from .simmods import *
from errConceal.caltec import *
from errConceal.altec import *
from errConceal.tc_algos import *
import cv2 as cv2
from PIL import Image
# ---------------------------------------------------------------------------- #
def fnRunImgClassDemo(modelDict,splitLayerDict,ecDict,batch_size,path_base,transDict,outputDir):
    print('TensorFlow version')
    print(tf.__version__)

    model_path = modelDict['fullModel']
    customObjects = modelDict['customObjects']
    task = modelDict['task']
    normalize = modelDict['normalize']
    reshapeDims = modelDict['reshapeDims']

    splitLayer = splitLayerDict['split']
    mobile_model_path = splitLayerDict['MobileModel']
    cloud_model_path = splitLayerDict['CloudModel']

    rowsPerPacket = transDict['rowsperpacket']
    quantization  = transDict['quantization']
    numberOfBits_1 = quantization[1]['numberOfBits']
    numberOfBits_2 = quantization[2]['numberOfBits']
    channel = transDict['channel']

    res_data_dir = outputDir['resDataDir'] # directory for loss maps.
    sim_data_dir = outputDir['simDataDir'] # directory for simulation results.
    # ------------------------------------------------------------------------ #
    # tensorflow.keras deep model loading.
    loaded_model = tf.keras.models.load_model(os.path.join(model_path))
    loaded_model_config = loaded_model.get_config()
    loaded_model_name = loaded_model_config['name']

    # Check if mobile and cloud sub-models are already available:
    if os.path.isfile(mobile_model_path) and os.path.isfile(cloud_model_path):
        print(f'Sub-models of {loaded_model_name} split at {splitLayer} are available.')
        mobile_model = tf.keras.models.load_model(os.path.join(mobile_model_path))
        cloud_model = tf.keras.models.load_model(os.path.join(cloud_model_path))
    else:
        # if not, split the deep model.
        # Object for splitting a tf.keras model into a mobile sub-model and a cloud
        # sub-model at the chosen split layer 'splitLayer'.
        testModel = BrokenModel(loaded_model, splitLayer, customObjects)
        testModel.splitModel()
        mobile_model = testModel.deviceModel
        cloud_model = testModel.remoteModel

        # Save the mobile and cloud sub-model
        mobile_model.save(mobile_model_path)
        cloud_model.save(cloud_model_path)
    # ---------------------------------------------------------------------------- #
    # Create results directory
    if 'GilbertChannel' in channel:
        lossProbability = channel['GilbertChannel']['lossProbability']
        burstLength = channel['GilbertChannel']['burstLength']
        results_dir = os.path.join(sim_data_dir,path_base,loaded_model_name,'demo',splitLayer+'_lp_'+str(lossProbability)+'_Bl_'+str(burstLength))
        channel_flag = 'GC'
    elif 'RandomLossChannel' in channel:
        lossProbability = channel['RandomLossChannel']['lossProbability']
        results_dir = os.path.join(sim_data_dir,path_base,loaded_model_name,'demo',splitLayer+'_lp_'+str(lossProbability))
        channel_flag = 'RL'
    elif 'ExternalChannel' in channel:
        print('External packet traces imported')
        results_dir = os.path.join(sim_data_dir,path_base,loaded_model_name,'demo',splitLayer+'_ext_trace')
        channel_flag = 'EX'
        num_channels = transDict['channel']['ExternalChannel']['num_channels']
        ext_dir = os.path.join(res_data_dir,path_base,loaded_model_name,splitLayer)
    else:
        # No lossy channel. This means we are doing a quantization experiment.
        channel_flag = 'NC'
        results_dir = os.path.join(sim_data_dir,path_base,loaded_model_name,'demo',splitLayer+'_NoChannel')
        MC_runs = [0,1] # with no lossy channel, there's no need to do monte carlo runs because each monte carlo run would give the same results.

    if channel_flag in ['GC','RL','EX']:
        # Only load altec weights if we will be doing error concealment.
        tc_weights_path = ecDict['ALTeC']['weightspath']
        altec_w_path = os.path.join(tc_weights_path,loaded_model_name,splitLayer,splitLayer+'_rpp_'+str(rowsPerPacket)+'_'+str(numberOfBits_1)+'Bits_tensor_weights.npy')
        altec_pkt_w = np.load(altec_w_path)
        print(f'Loaded ALTeC weights for splitLayer {splitLayer} and {rowsPerPacket} rows per packet. Shape {np.shape(altec_pkt_w)}')
        halrtc_iters = ecDict['HaLRTC']['numiters']
        silrtc_iters = ecDict['SiLRTC']['numiters']
        inpaint_radius = ecDict['InpaintNS']['radius']

    os.makedirs(results_dir,exist_ok=True)
    res_filename = '_'+str(numberOfBits_1)+'Bits_'+str(numberOfBits_2)+'Bits_'
    # ------------------------------------------------------------------------ #
    # Objects for the channel, quantization.
    if channel_flag != 'EX':
        channel = loadChannel(channel)
    quant_tensor1 = quantInit(quantization,tensor_id = 1)
    quant_tensor2 = quantInit(quantization,tensor_id = 2)
    # ------------------------------------------------------------------------ #
    # Load the dataset
    dataset_x_files,dataset_y_labels,file_names = fn_Data_PreProcessing_ImgClass(path_base,reshapeDims,normalize)
    # ------------------------------------------------------------------------ #
    # Process the dataset.
    batched_y_labels = [dataset_y_labels[i:i + batch_size] for i in range(0, len(dataset_y_labels), batch_size)]
    batched_x_files = [dataset_x_files[i: i + batch_size] for i in range(0,len(dataset_x_files),batch_size)]

    if channel_flag == 'EX':
        loss_matrix_mc = []
        print('Loading external packet traces')
        for i_mc in range(MC_runs[0],MC_runs[1]):
            # Load external packet traces as loss matrices.
            lossMap_list = []
            for i_c in range(num_channels):
                df = pd.read_excel(os.path.join(ext_dir,'Rpp_'+str(rowsPerPacket)+'_MC_'+str(i_mc)+'.xlsx'),sheet_name=[str(i_c)],engine='openpyxl')
                lossMap_channel = (df[str(i_c)].to_numpy())[:,1:].astype(np.bool)
                lossMap_list.append(lossMap_channel)

            loss_matrix_all = np.dstack(lossMap_list)
            loss_matrix_ex  = [loss_matrix_all[k_batch:k_batch+batch_size,:,:] for k_batch in range(0,np.shape(loss_matrix_all)[0],batch_size)]
            loss_matrix_mc.append(loss_matrix_ex)

    # lists to store results.
    true_labels = []
    top1_pred_full_model = []
    top1_pred_split_model = []
    top5_pred_full_model = []
    top5_pred_split_model = []
    top1_pred_caltec = []
    top5_pred_caltec = []
    top1_pred_altec = []
    top5_pred_altec = []
    top1_pred_halrtc = []
    top5_pred_halrtc = []
    top1_pred_silrtc = []
    top5_pred_silrtc = []
    top1_pred_inpaint = []
    top5_pred_inpaint = []

    top1_conf_full = []
    top1_conf_split = []
    top1_conf_caltec = []
    top1_conf_altec = []
    top1_conf_halrtc = []
    top1_conf_silrtc = []
    top1_conf_inpaint = []

    for i_b in range(len(batched_y_labels)):
        # Run through Monte Carlo experiments through each batch.
        print(f"Batch {i_b}")
        batch_labels = np.asarray(batched_y_labels[i_b],dtype=np.int64)
        true_labels.extend(batch_labels)
        batch_imgs = batched_x_files[i_b]
        batch_imgs_stacked = np.vstack([i[np.newaxis,...] for i in batch_imgs])
        # ---------------------------------------------------------------- #
        full_model_out = loaded_model.predict(batch_imgs_stacked)
        batch_top1_predictions = np.argmax(full_model_out,axis=1)
        batch_confidence = np.max(full_model_out,axis=1)
        top1_pred_full_model.extend(batch_top1_predictions)
        top1_conf_full.extend(batch_confidence)

        for i_item in range(np.shape(full_model_out)[0]):
            item_top5_predictions = np.argpartition(-full_model_out[i_item,:],5)[:5]
            top5_pred_full_model.append(item_top5_predictions)
         # --------------------------------------------------------------- #
        deviceOut = mobile_model.predict(batch_imgs_stacked)
        print(f'Shape of device out tensor {np.shape(deviceOut)}')
        # ---------------------------------------------------------------- #
        devOut = []
        if not isinstance(deviceOut, list):
            devOut.append(deviceOut)
            deviceOut = devOut
            # deviceOut is the output tensor for a batch of data.

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

        # Save quantized tensors as image.
        for i in range(len(deviceOut)):
            quant_tensor = deviceOut[i]
            for item_index in range(np.shape(quant_tensor)[0]):
                for i_c in range(np.shape(quant_tensor)[-1]):
                    tensor_channel = Image.fromarray(quant_tensor[item_index,:,:,i_c].astype(np.uint8))
                    tensor_channel.save(os.path.join(results_dir,'original_batch_'+str(i_b)+'_item_'+str(item_index)+'_tensor_'+str(i)+'_channel_'+str(i_c)+res_filename+'.png'))
        # -------------------------------------------------------------------- #
        # Transmit the tensor deviceOut through the channel.
        if channel_flag in ['GC','RL']:
            # if a lossy channel has to be realized.
            # if mc_task == 'GenLossPatterns':
            # if we want to generate packet loss patterns.
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
            deviceOut = dOut
            # ---------------------------------------------------------------- #
            # packetize tensor.
            pkt_obj_list = []
            for i in range(len(deviceOut)):
                pkt_obj_list.append(PacketModel(rows_per_packet=rowsPerPacket,data_tensor=np.copy(deviceOut[i].data_tensor)))
        # -------------------------------------------------------------------- #
        if channel_flag == 'EX':
            batch_loss_matrix = loss_matrix_mc[i_mc]
            loss_matrix = [batch_loss_matrix[i_b]]
        # -------------------------------------------------------------------- #
        if channel_flag in ['GC','RL','EX']:
            # ---------------------------------------------------------------- #
            # apply the loss matrix to the tensor.
            for i in range(len(pkt_obj_list)):
                loss_map = lossMatrix[i]
                #print(np.shape(loss_map))
                channel_width = np.shape(pkt_obj_list[i].packet_seq)[3]

                # loop through items in batch.
                for item_index in range(np.shape(loss_map)[0]):
                    item_lost_map = loss_map[item_index,:,:]
                    lost_pkt_indices,lost_channel_indices = np.where(item_lost_map == False)
                    if len(lost_pkt_indices) != 0:
                        # drop packet in tensor.
                        for k in range(len(lost_pkt_indices)):
                            pkt_obj_list[i].packet_seq[item_index,lost_pkt_indices[k],:,:,lost_channel_indices[k]] = np.zeros([rowsPerPacket,channel_width])
            for i in range(len(deviceOut)):
                quant_tensor = pkt_obj_list[i].data_tensor
                for item_index in range(np.shape(quant_tensor)[0]):
                    for i_c in range(np.shape(quant_tensor)[-1]):
                        tensor_channel = Image.fromarray(quant_tensor[item_index,:,:,i_c].astype(np.uint8))
                        tensor_channel.save(os.path.join(results_dir,'corrupted_batch_'+str(i_b)+'_item_'+str(item_index)+'_tensor_'+str(i)+'_channel_'+str(i_c)+res_filename+'.png'))
            deviceOut = pkt_obj_list
        # --------------------------------------------------====-------------- #
        # Inverse quantize received packets.
        # If necessary, inverse quantize tensors.
        if len(deviceOut) > 1:
            # If more than one tensor is transmitted from the mobile device to the cloud.
            if quant_tensor1!= 'noQuant':
                print("Inverse quantizing tensors")
                if channel_flag != 'NC':
                    quant_tensor1.quanData = deviceOut[0].data_tensor
                    qMin, qMax = quanParams_1
                    quant_tensor1.min = qMin
                    quant_tensor1.max = qMax
                    deviceOut[0].data_tensor = quant_tensor1.inverseQuantizer()

                    quant_tensor2.quanData = deviceOut[1].data_tensor
                    qMin, qMax = quanParams_2
                    quant_tensor2.min = qMin
                    quant_tensor2.max = qMax
                    deviceOut[1].data_tensor = quant_tensor2.inverseQuantizer()
                else:
                    # no channel.
                    quant_tensor1.quanData = deviceOut[0]
                    qMin, qMax = quanParams_1
                    quant_tensor1.min = qMin
                    quant_tensor1.max = qMax
                    deviceOut[0] = quant_tensor1.inverseQuantizer()

                    quant_tensor2.quanData = deviceOut[1]
                    qMin, qMax = quanParams_2
                    quant_tensor2.min = qMin
                    quant_tensor2.max = qMax
                    deviceOut[1] = quant_tensor2.inverseQuantizer()
        else:
            # A single tensor is transmitted from the mobile device to the cloud.
            if quant_tensor1 != 'noQuant':
                print("Inverse quantizing tensor")
                if channel_flag != 'NC':
                    # we have lossy channels (either GE, RL or external packet traces.)
                    quant_tensor1.quanData = deviceOut[0].data_tensor
                    qMin, qMax = quanParams_1
                    quant_tensor1.min = qMin
                    quant_tensor1.max = qMax
                    deviceOut[0].data_tensor = quant_tensor1.inverseQuantizer()
                else:
                    # no channel.
                    quant_tensor1.quanData = deviceOut[0]
                    qMin, qMax = quanParams_1
                    quant_tensor1.min = qMin
                    quant_tensor1.max = qMax
                    deviceOut[0] = quant_tensor1.inverseQuantizer()
        cOut = []
        for i in range(len(deviceOut)):
            if channel_flag != 'NC':
                cOut.append(np.copy(deviceOut[i].data_tensor))
            else:
                cOut.append(np.copy(deviceOut[i]))

        deviceOut = cOut
        # -------------------------------------------------------------------- #
        # Run cloud prediction on channel output data.
        tensor_out = cloud_model.predict(deviceOut)
        cloud_Top1_pred = np.argmax(tensor_out,axis=1)
        cloud_Top1_confidence = np.max(tensor_out,axis=1)
        top1_pred_split_model.extend(cloud_Top1_pred)
        top1_conf_split.extend(cloud_Top1_confidence)

        for i_item in range(np.shape(tensor_out)[0]):
            item_top5_predictions = np.argpartition(-tensor_out[i_item,:],5)[:5]
            top5_pred_split_model.append(item_top5_predictions)
        # -------------------------------------------------------------------- #
        # Run packet loss concealment methods if a lossy channel was used.
        if channel_flag in ['EX','RL','GC']:
            # Flush missing packets out of tensor.
            # packetize tensor.
            pkt_obj_list_caltec = []
            pkt_obj_list_altec = []
            pkt_obj_list_halrtc = []
            pkt_obj_list_silrtc = []
            pkt_obj_list_inpaint = []
            inpaint_masks_list = []
            for i in range(len(deviceOut)):
                pkt_obj_list_caltec.append(PacketModel(rows_per_packet=rowsPerPacket,data_tensor=np.copy(deviceOut[i])))
                pkt_obj_list_altec.append(PacketModel(rows_per_packet=rowsPerPacket,data_tensor=np.copy(deviceOut[i])))
                pkt_obj_list_halrtc.append(PacketModel(rows_per_packet=rowsPerPacket,data_tensor=np.copy(deviceOut[i])))
                pkt_obj_list_silrtc.append(PacketModel(rows_per_packet=rowsPerPacket,data_tensor=np.copy(deviceOut[i])))
                pkt_obj_list_inpaint.append(PacketModel(rows_per_packet=rowsPerPacket,data_tensor=np.copy(deviceOut[i])))
                inpaint_masks = np.zeros(np.shape(pkt_obj_list[i].data_tensor),dtype= np.uint8)
                inpaint_masks_list.append(PacketModel(rows_per_packet=rowsPerPacket,data_tensor=np.copy(inpaint_masks)))
            # ---------------------------------------------------------------- #
            # apply this loss matrix to the tensor.
            for i in range(len(pkt_obj_list)):
                loss_map = lossMatrix[i]
                channel_width = np.shape(pkt_obj_list_caltec[i].packet_seq)[3]

                # loop through items in batch.
                for item_index in range(np.shape(loss_map)[0]):
                    item_lost_map = loss_map[item_index,:,:]
                    lost_pkt_indices,lost_channel_indices = np.where(item_lost_map == False)
                    if len(lost_pkt_indices) != 0:
                        # drop packet in tensor.
                        for k in range(len(lost_pkt_indices)):
                            pkt_obj_list_caltec[i].packet_seq[item_index,lost_pkt_indices[k],:,:,lost_channel_indices[k]] = np.zeros([rowsPerPacket,channel_width])
                            pkt_obj_list_altec[i].packet_seq[item_index,lost_pkt_indices[k],:,:,lost_channel_indices[k]] = np.zeros([rowsPerPacket,channel_width])
                            pkt_obj_list_halrtc[i].packet_seq[item_index,lost_pkt_indices[k],:,:,lost_channel_indices[k]] = np.zeros([rowsPerPacket,channel_width])
                            pkt_obj_list_silrtc[i].packet_seq[item_index,lost_pkt_indices[k],:,:,lost_channel_indices[k]] = np.zeros([rowsPerPacket,channel_width])
                            pkt_obj_list_inpaint[i].packet_seq[item_index,lost_pkt_indices[k],:,:,lost_channel_indices[k]] = np.zeros([rowsPerPacket,channel_width])
                            inpaint_masks_list[i].packet_seq[item_index,lost_pkt_indices[k],:,:,lost_channel_indices[k]] = 255*np.ones([rowsPerPacket,channel_width])
            # ---------------------------------------------------------------- #
            # Error loss concealment.
            print('Running packet loss concealment.')
            for i in range(len(pkt_obj_list_caltec)):
                loss_map = lossMatrix[i]
                # ALTeC
                repaired_tensor = fn_complete_tensor_altec_pkts_star(np.copy(pkt_obj_list_altec[i].data_tensor),rowsPerPacket,altec_pkt_w,loss_map)
                pkt_obj_list_altec[i].data_tensor = repaired_tensor
                # CALTeC, HaLRTC and SiLRTC
                for item_index in range(len(batch_labels)):
                    pkt_obj_list_caltec[i] = fn_caltec(loss_map,pkt_obj_list_caltec[i],item_index)
                    # loop through items in batch.
                    X_damaged = pkt_obj_list_halrtc[i].data_tensor[item_index,:,:,:]
                    lost_map = loss_map[item_index,:,:]
                    received_map = np.zeros(np.shape(pkt_obj_list_halrtc[i].data_tensor[item_index,:,:,:]),dtype=bool)
                    for pkt_index in range(np.shape(lost_map)[0]):
                        for channel_index in range(np.shape(lost_map)[1]):
                            if lost_map[pkt_index,channel_index] == True:
                                received_map[pkt_index*rowsPerPacket:(pkt_index+1)*rowsPerPacket,:,channel_index] = np.ones([rowsPerPacket,channel_width],dtype=bool)

                    lost_pkt_indices,lost_channel_indices = np.where(lost_map == False)
                    inpaint_masks_tensor = inpaint_masks_list[i].data_tensor[item_index,:,:,:]

                    for i_c in range(np.shape(X_damaged)[-1]):
                        if i_c in lost_channel_indices:
                            damaged_channel = X_damaged[:,:,i_c]
                            mask_channel = inpaint_masks_tensor[:,:,i_c]
                            pkt_obj_list_inpaint[i].data_tensor[item_index,:,:,i_c] = cv2.inpaint(np.copy(damaged_channel),np.copy(mask_channel),inpaint_radius,cv2.INPAINT_NS)

                    subs = np.argwhere(received_map == True)
                    vals = list(map(lambda x, y, z: X_damaged[x][y][z], subs[:,0], subs[:,1], subs[:,2]))

                    X_estimated_silrtc, error_iters_tc = fn_silrtc_damaged_error(np.copy(X_damaged),silrtc_iters,subs,vals)
                    X_estimated_halrtc, error_iters_tc = fn_halrtc_damaged_error(np.copy(X_damaged),halrtc_iters,subs,vals)

                    pkt_obj_list_silrtc[i].data_tensor[item_index,:,:,:] = X_estimated_silrtc[:,:,:,-1]
                    pkt_obj_list_halrtc[i].data_tensor[item_index,:,:,:] = X_estimated_halrtc[:,:,:,-1]

            caltec_quant = []
            altec_quant = []
            halrtc_quant = []
            silrtc_quant = []
            inpaint_quant = []
            if quant_tensor1!= 'noQuant':
                print("Quantizing tensor.")
                quant_tensor1.bitQuantizer(np.copy(pkt_obj_list_caltec[0].data_tensor))
                caltec_quant.append(quant_tensor1.quanData)
                quant_tensor1.bitQuantizer(np.copy(pkt_obj_list_altec[0].data_tensor))
                altec_quant.append(quant_tensor1.quanData)
                quant_tensor1.bitQuantizer(np.copy(pkt_obj_list_halrtc[0].data_tensor))
                halrtc_quant.append(quant_tensor1.quanData)
                quant_tensor1.bitQuantizer(np.copy(pkt_obj_list_silrtc[0].data_tensor))
                silrtc_quant.append(quant_tensor1.quanData)
                quant_tensor1.bitQuantizer(np.copy(pkt_obj_list_inpaint[0].data_tensor))
                inpaint_quant.append(quant_tensor1.quanData)

            # Save quantized tensors as image.
            for i in range(len(deviceOut)):
                quant_caltec = caltec_quant[i]
                quant_altec = altec_quant[i]
                quant_halrtc = halrtc_quant[i]
                quant_silrtc = silrtc_quant[i]
                quant_inpaint = inpaint_quant[i]
                for item_index in range(np.shape(quant_caltec)[0]):
                    for i_c in range(np.shape(quant_caltec)[-1]):
                        tensor_channel = Image.fromarray(quant_caltec[item_index,:,:,i_c].astype(np.uint8))
                        tensor_channel.save(os.path.join(results_dir,'CALTeC_batch_'+str(i_b)+'_item_'+str(item_index)+'_tensor_'+str(i)+'_channel_'+str(i_c)+res_filename+'.png'))

                        tensor_channel = Image.fromarray(quant_altec[item_index,:,:,i_c].astype(np.uint8))
                        tensor_channel.save(os.path.join(results_dir,'ALTeC_batch_'+str(i_b)+'_item_'+str(item_index)+'_tensor_'+str(i)+'_channel_'+str(i_c)+res_filename+'.png'))

                        tensor_channel = Image.fromarray(quant_halrtc[item_index,:,:,i_c].astype(np.uint8))
                        tensor_channel.save(os.path.join(results_dir,'HaLRTC_batch_'+str(i_b)+'_item_'+str(item_index)+'_tensor_'+str(i)+'_channel_'+str(i_c)+res_filename+'.png'))

                        tensor_channel = Image.fromarray(quant_silrtc[item_index,:,:,i_c].astype(np.uint8))
                        tensor_channel.save(os.path.join(results_dir,'SiLRTC_batch_'+str(i_b)+'_item_'+str(item_index)+'_tensor_'+str(i)+'_channel_'+str(i_c)+res_filename+'.png'))

                        tensor_channel = Image.fromarray(quant_inpaint[item_index,:,:,i_c].astype(np.uint8))
                        tensor_channel.save(os.path.join(results_dir,'InpaintNS_batch_'+str(i_b)+'_item_'+str(item_index)+'_tensor_'+str(i)+'_channel_'+str(i_c)+res_filename+'.png'))
            # ---------------------------------------------------------------- #
            caltec_tensor_list = []
            altec_tensor_list = []
            halrtc_tensor_list = []
            silrtc_tensor_list = []
            inpaint_tensor_list = []

            for i in range(len(pkt_obj_list_caltec)):
                caltec_tensor_list.append(pkt_obj_list_caltec[i].data_tensor)
                altec_tensor_list.append(pkt_obj_list_altec[i].data_tensor)
                halrtc_tensor_list.append(pkt_obj_list_halrtc[i].data_tensor)
                silrtc_tensor_list.append(pkt_obj_list_silrtc[i].data_tensor)
                inpaint_tensor_list.append(pkt_obj_list_inpaint[i].data_tensor)

            caltec_out = cloud_model.predict(caltec_tensor_list)
            caltec_Top1_pred = np.argmax(caltec_out,axis=1)
            caltec_Top1_confidence = np.max(caltec_out,axis=1)
            top1_conf_caltec.extend(caltec_Top1_confidence)
            top1_pred_caltec.extend(caltec_Top1_pred)

            for i_item in range(np.shape(caltec_out)[0]):
                item_top5_predictions = np.argpartition(-caltec_out[i_item,:],5)[:5]
                top5_pred_caltec.append(item_top5_predictions)

            altec_out = cloud_model.predict(altec_tensor_list)
            altec_Top1_pred = np.argmax(altec_out,axis=1)
            altec_Top1_confidence = np.max(altec_out,axis=1)
            top1_conf_altec.extend(altec_Top1_confidence)
            top1_pred_altec.extend(altec_Top1_pred)

            for i_item in range(np.shape(altec_out)[0]):
                item_top5_predictions = np.argpartition(-altec_out[i_item,:],5)[:5]
                top5_pred_altec.append(item_top5_predictions)

            halrtc_out = cloud_model.predict(halrtc_tensor_list)
            halrtc_Top1_pred = np.argmax(halrtc_out,axis=1)
            halrtc_Top1_confidence = np.max(halrtc_out,axis=1)
            top1_conf_halrtc.extend(halrtc_Top1_confidence)
            top1_pred_halrtc.extend(halrtc_Top1_pred)

            for i_item in range(np.shape(halrtc_out)[0]):
                item_top5_predictions = np.argpartition(-halrtc_out[i_item,:],5)[:5]
                top5_pred_halrtc.append(item_top5_predictions)

            silrtc_out = cloud_model.predict(silrtc_tensor_list)
            silrtc_Top1_pred = np.argmax(silrtc_out,axis=1)
            silrtc_Top1_confidence = np.max(silrtc_out,axis=1)
            top1_conf_silrtc.extend(silrtc_Top1_confidence)
            top1_pred_silrtc.extend(silrtc_Top1_pred)

            for i_item in range(np.shape(silrtc_out)[0]):
                item_top5_predictions = np.argpartition(-silrtc_out[i_item,:],5)[:5]
                top5_pred_silrtc.append(item_top5_predictions)

            inpaint_out = cloud_model.predict(inpaint_tensor_list)
            inpaint_Top1_pred = np.argmax(inpaint_out,axis=1)
            inpaint_Top1_confidence = np.max(inpaint_out,axis=1)
            top1_conf_inpaint.extend(inpaint_Top1_confidence)
            top1_pred_inpaint.extend(inpaint_Top1_pred)

            for i_item in range(np.shape(inpaint_out)[0]):
                item_top5_predictions = np.argpartition(-inpaint_out[i_item,:],5)[:5]
                top5_pred_inpaint.append(item_top5_predictions)

    if channel_flag in ['GC','RL','EX']:
        f = open(os.path.join(results_dir,'summary_'+channel_flag+'.txt'),"w")
        f.write('Summary of results over lossy channels \n')
        f.write('True classes \n')
        f.write(str(true_labels))
        f.write('\n Top-1 No Channel predictions \n')
        f.write(str(top1_pred_full_model))
        f.write('\n Top-1 No Channel confidence \n')
        f.write(str(top1_conf_full))
        f.write('\n Top-5 No Channel predictions \n')
        f.write(str(top5_pred_full_model))
        f.write('\n Top-1 No Completion predictions \n')
        f.write(str(top1_pred_split_model))
        f.write('\n Top-1 No Completion confidence \n')
        f.write(str(top1_conf_split))
        f.write('\n Top-5 No Completion predictions \n')
        f.write(str(top5_pred_split_model))
        f.write('\n Top-1 CALTeC predictions \n')
        f.write(str(top1_pred_caltec))
        f.write('\n Top-1 CALTeC confidence \n')
        f.write(str(top1_conf_caltec))
        f.write('\n Top-5 CALTeC predictions \n')
        f.write(str(top5_pred_caltec))
        f.write('\n Top-1 ALTeC predictions \n')
        f.write(str(top1_pred_altec))
        f.write('\n Top-1 ALTeC confidence \n')
        f.write(str(top1_conf_altec))
        f.write('\n Top-5 ALTeC predictions \n')
        f.write(str(top5_pred_altec))
        f.write('\n Top-1 HaLRTC predictions \n')
        f.write(str(top1_pred_halrtc))
        f.write('\n Top-1 HaLRTC confidence \n')
        f.write(str(top1_conf_halrtc))
        f.write('\n Top-5 HaLRTC predictions \n')
        f.write(str(top5_pred_halrtc))
        f.write('\n Top-1 SiLRTC predictions \n')
        f.write(str(top1_pred_silrtc))
        f.write('\n Top-1 SiLRTC confidence \n')
        f.write(str(top1_conf_silrtc))
        f.write('\n Top-5 SiLRTC predictions \n')
        f.write(str(top5_pred_silrtc))
        f.write('\n Top-1 Inpaint Navier Stokes predictions \n')
        f.write(str(top1_pred_inpaint))
        f.write('\n Top-1 Inpaint confidence \n')
        f.write(str(top1_conf_inpaint))
        f.write('\n Top-5 Inpaint Navier Stokes predictions \n')
        f.write(str(top5_pred_inpaint))
        f.close()
