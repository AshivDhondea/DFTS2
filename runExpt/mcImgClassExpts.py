"""Run Monte Carlo image classification experiments.
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
# ---------------------------------------------------------------------------- #
def fnRunImgClassMC(modelDict,splitLayerDict,ecDict,MC_runs,mc_task,batch_size,path_base,transDict,outputDir):
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
        results_dir = os.path.join(sim_data_dir,path_base,loaded_model_name,splitLayer+'_lp_'+str(lossProbability)+'_Bl_'+str(burstLength))
        channel_flag = 'GC'
        loss_maps_dir = os.path.join(res_data_dir,path_base,loaded_model_name,splitLayer+'_lp_'+str(lossProbability)+'_Bl_'+str(burstLength))
        os.makedirs(loss_maps_dir,exist_ok=True)
    elif 'RandomLossChannel' in channel:
        lossProbability = channel['RandomLossChannel']['lossProbability']
        results_dir = os.path.join(sim_data_dir,path_base,loaded_model_name,splitLayer+'_lp_'+str(lossProbability))
        channel_flag = 'RL'
        loss_maps_dir = os.path.join(res_data_dir,path_base,loaded_model_name,splitLayer+'_lp_'+str(lossProbability))
        os.makedirs(loss_maps_dir,exist_ok=True)
    elif 'ExternalChannel' in channel:
        print('External packet traces imported')
        results_dir = os.path.join(sim_data_dir,path_base,loaded_model_name,splitLayer)
        channel_flag = 'EX'
        num_channels = transDict['channel']['ExternalChannel']['num_channels']
        ext_dir = os.path.join(res_data_dir,path_base,loaded_model_name,splitLayer)

    else:
        # No lossy channel. This means we are doing a quantization experiment.
        channel_flag = 'NC'
        results_dir = os.path.join(sim_data_dir,path_base,loaded_model_name,splitLayer+'_NoChannel')
        MC_runs = [0,1] # with no lossy channel, there's no need to do monte carlo runs because each monte carlo run would give the same results.

    if ecDict != 'noEC':
        ec_method = [*ecDict][0]
        if ec_method == 'ALTeC':
            tc_weights_path = ecDict['ALTeC']['weightspath']
            altec_w_path = os.path.join(tc_weights_path,loaded_model_name,splitLayer,splitLayer+'_rpp_'+str(rowsPerPacket)+'_'+str(numberOfBits_1)+'Bits_tensor_weights.npy')
            altec_pkt_w = np.load(altec_w_path)
            print(f'Loaded ALTeC weights for splitLayer {splitLayer} and {rowsPerPacket} rows per packet. Shape {np.shape(altec_pkt_w)}')

    os.makedirs(results_dir,exist_ok=True)
    # ------------------------------------------------------------------------ #
    # Plot architecture of full model. Save its summary as txt file.
    tf.keras.utils.plot_model(loaded_model,to_file=os.path.join(sim_data_dir,path_base,loaded_model_name,splitLayer+'_full_model.png'),show_shapes=True)
    with open(os.path.join(sim_data_dir,path_base,loaded_model_name,splitLayer+'_full_model.txt'),'w') as fh:
        loaded_model.summary(print_fn = lambda x: fh.write(x + '\n'))

    # Plot architecture of mobile and cloud sub models. Save their summary as txt file.
    tf.keras.utils.plot_model(mobile_model,to_file=os.path.join(sim_data_dir,path_base,loaded_model_name,splitLayer+'_mobile_model.png'),show_shapes=True)

    with open(os.path.join(sim_data_dir,path_base,loaded_model_name,splitLayer+'_mobile_model.txt'),'w') as fh:
        mobile_model.summary(print_fn = lambda x: fh.write(x + '\n'))

    tf.keras.utils.plot_model(cloud_model,to_file=os.path.join(sim_data_dir,path_base,loaded_model_name,splitLayer+'_cloud_model.png'),show_shapes=True)

    with open(os.path.join(sim_data_dir,path_base,loaded_model_name,splitLayer+'_cloud_model.txt'),'w') as fh:
        cloud_model.summary(print_fn = lambda x: fh.write(x + '\n'))
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

    num_MC = MC_runs[1]-MC_runs[0]
    mc_cloud_top1_accuracy = np.zeros([num_MC],dtype=np.float64)
    mc_full_top1_accuracy = np.zeros([num_MC],dtype=np.float64)
    mc_cloud_top5_accuracy = np.zeros([num_MC],dtype=np.float64)
    mc_full_top5_accuracy = np.zeros([num_MC],dtype=np.float64)
    mc_repaired_top1_accuracy = np.zeros([num_MC],dtype=np.float64)
    mc_repaired_top5_accuracy = np.zeros([num_MC],dtype=np.float64)
    mc_time = np.zeros([num_MC],dtype=np.float64)

    if mc_task == 'LoadLossPatterns':
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

    for i_mc in range(MC_runs[0],MC_runs[1]):
        # lists to store results.
        true_labels = []
        top1_pred_full_model = []
        top1_pred_split_model = []
        top5_pred_full_model = []
        top5_pred_split_model = []
        top1_pred_repaired = []
        top5_pred_repaired = []

        for i_b in range(len(batched_y_labels)):
            # Run through Monte Carlo experiments through each batch.
            print(f"Monte Carlo run {i_mc} on batch {i_b}")
            batch_labels = np.asarray(batched_y_labels[i_b],dtype=np.int64)
            true_labels.extend(batch_labels)
            batch_imgs = batched_x_files[i_b]
            batch_imgs_stacked = np.vstack([i[np.newaxis,...] for i in batch_imgs])
            # ---------------------------------------------------------------- #
            full_model_out = loaded_model.predict(batch_imgs_stacked)
            batch_top1_predictions = np.argmax(full_model_out,axis=1)
            batch_confidence = np.max(full_model_out,axis=1)
            top1_pred_full_model.extend(batch_top1_predictions)

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
            # ---------------------------------------------------------------- #
            # Transmit the tensor deviceOut through the channel.
            if channel_flag in ['GC','RL']:
                # if a lossy channel has to be realized.
                if mc_task == 'GenLossPatterns':
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
                        # Save loss map.
                        np.save(os.path.join(loss_maps_dir,'MC_'+str(i_mc)+'_batch_'+str(i_b)+'_tensor_'+str(i)+'_rpp_'+str(rowsPerPacket)+'.npy'),lM)
                    deviceOut = dOut
            # ---------------------------------------------------------------- #
            if mc_task == 'LoadLossPatterns':
                if channel_flag == 'EX':
                    print('Loading external packet traces')
                    batch_loss_matrix = loss_matrix_mc[i_mc]
                    loss_matrix = [batch_loss_matrix[i_b]]

                if channel_flag in ['GC','RL']:
                    # Load loss matrices from file.
                    loss_matrix = []
                    for i in range(len(deviceOut)):
                        loss_matrix.append(np.load(os.path.join(loss_maps_dir,'MC_'+str(i_mc)+'_batch_'+str(i_b)+'_tensor_'+str(i)+'_rpp_'+str(rowsPerPacket)+'.npy')))

                # Apply loss matrix (either from external trace or from previously GenLossPatterns experiment.)
                # packetize tensor.
                pkt_obj_list = []
                for i in range(len(deviceOut)):
                    pkt_obj_list.append(PacketModel(rows_per_packet=rowsPerPacket,data_tensor=deviceOut[i]))
                # -------------------------------------------------------- #
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
                                pkt_obj_list[i].packet_seq[item_index,lost_pkt_indices[k],:,:,lost_channel_indices[k]] = np.zeros([rowsPerPacket,channel_width])

                deviceOut = pkt_obj_list
            # ---------------------------------------------------------------- #
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
            # ---------------------------------------------------------------- #
            # Run cloud prediction on channel output data.
            tensor_out = cloud_model.predict(deviceOut)
            cloud_Top1_pred = np.argmax(tensor_out,axis=1)
            cloud_Top1_confidence = np.max(tensor_out,axis=1)
            top1_pred_split_model.extend(cloud_Top1_pred)

            for i_item in range(np.shape(tensor_out)[0]):
                item_top5_predictions = np.argpartition(-tensor_out[i_item,:],5)[:5]
                top5_pred_split_model.append(item_top5_predictions)
            # ---------------------------------------------------------------- #
            if ecDict != 'noEC':
                # Do error concealment with selected method.
                # ------------------------------------------------------------ #
                # Flush missing packets out of tensor.
                # packetize tensor.
                pkt_obj_list = []
                inpaint_masks_list = []
                for i in range(len(deviceOut)):
                    pkt_obj_list.append(PacketModel(rows_per_packet=rowsPerPacket,data_tensor=deviceOut[i]))
                    inpaint_masks = np.zeros(np.shape(pkt_obj_list[i].data_tensor),dtype= np.uint8)
                    inpaint_masks_list.append(PacketModel(rows_per_packet=rowsPerPacket,data_tensor=np.copy(inpaint_masks)))
                # ------------------------------------------------------------ #
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
                                pkt_obj_list[i].packet_seq[item_index,lost_pkt_indices[k],:,:,lost_channel_indices[k]] = np.zeros([rowsPerPacket,channel_width])
                                inpaint_masks_list[i].packet_seq[item_index,lost_pkt_indices[k],:,:,lost_channel_indices[k]] = 255*np.ones([rowsPerPacket,channel_width])
                # ------------------------------------------------------------ #
                # Find the selected method, and its params if necessary. Then do error concealment.
                ec_method = [*ecDict][0] # Find the selected error concealment method.
                mc_start_time = timer() # start timer.
                if ec_method == 'CALTeC':
                    for i in range(len(pkt_obj_list)):
                        loss_map = loss_matrix[i]
                        for item_index in range(np.shape(loss_map)[0]):
                            pkt_obj_list[i] = fn_caltec(loss_map,pkt_obj_list[i],item_index)

                if ec_method == 'ALTeC':
                    for i in range(len(pkt_obj_list)):
                        loss_map = loss_matrix[i]
                        repaired_tensor = fn_complete_tensor_altec_pkts_star(np.copy(pkt_obj_list[i].data_tensor),rowsPerPacket,altec_pkt_w,loss_map)
                        pkt_obj_list[i].data_tensor = repaired_tensor

                if ec_method == 'SiLRTC' or ec_method == 'HaLRTC':
                    num_iters = ecDict[ec_method]['numiters']
                    for i in range(len(pkt_obj_list)):
                        loss_map = loss_matrix[i]
                        for item_index in range(len(batch_labels)):
                            # loop through items in batch.
                            X_damaged = pkt_obj_list[i].data_tensor[item_index,:,:,:]
                            lost_map = loss_map[item_index,:,:]
                            received_map = np.zeros(np.shape(pkt_obj_list[i].data_tensor[item_index,:,:,:]),dtype=bool)
                            for pkt_index in range(np.shape(lost_map)[0]):
                                for channel_index in range(np.shape(lost_map)[1]):
                                    if lost_map[pkt_index,channel_index] == True:
                                        received_map[pkt_index*rowsPerPacket:(pkt_index+1)*rowsPerPacket,:,channel_index] = np.ones([rowsPerPacket,channel_width],dtype=bool)

                            subs = np.argwhere(received_map == True)
                            vals = list(map(lambda x, y, z: X_damaged[x][y][z], subs[:,0], subs[:,1], subs[:,2]))
                            if ec_method == 'SiLRTC':
                                X_estimated_tc, error_iters_tc = fn_silrtc_damaged_error(X_damaged,num_iters,subs,vals)

                            if ec_method == 'HaLRTC':
                                X_estimated_tc, error_iters_tc = fn_halrtc_damaged_error(X_damaged,num_iters,subs,vals)

                            pkt_obj_list[i].data_tensor[item_index,:,:,:] = X_estimated_tc[:,:,:,-1]

                if ec_method == 'InpaintNS':
                    inpaint_radius = ecDict[ec_method]['radius']
                    for i in range(len(pkt_obj_list)):
                        channel_width = np.shape(pkt_obj_list[i].data_tensor)[2]
                        loss_map = loss_matrix[i]

                        for item_index in range(len(batch_labels)):
                            item_lost_map = loss_map[item_index,:,:]
                            lost_pkt_indices,lost_channel_indices = np.where(item_lost_map == False)
                            # loop through items in batch.
                            X_damaged = pkt_obj_list[i].data_tensor[item_index,:,:,:]
                            inpaint_masks_tensor = inpaint_masks_list[i].data_tensor[item_index,:,:,:]

                            for i_c in range(np.shape(X_damaged)[-1]):
                                if i_c in lost_channel_indices:
                                    damaged_channel = X_damaged[:,:,i_c]
                                    mask_channel = inpaint_masks_tensor[:,:,i_c]
                                    pkt_obj_list[i].data_tensor[item_index,:,:,i_c] = cv2.inpaint(np.copy(damaged_channel),np.copy(mask_channel),inpaint_radius,cv2.INPAINT_NS)
                mc_end_time = timer() # end timer.
                mc_time[i_mc - MC_runs[0]] += (mc_end_time - mc_start_time)
                # ------------------------------------------------------------ #
                repaired_tensor_list = []
                for i in range(len(pkt_obj_list)):
                    repaired_tensor_list.append(pkt_obj_list[i].data_tensor)

                tensor_out = cloud_model.predict(repaired_tensor_list)
                repaired_Top1_pred = np.argmax(tensor_out,axis=1)
                repaired_Top1_confidence = np.max(tensor_out,axis=1)
                top1_pred_repaired.extend(repaired_Top1_pred)

                for i_item in range(np.shape(tensor_out)[0]):
                    item_top5_predictions = np.argpartition(-tensor_out[i_item,:],5)[:5]
                    top5_pred_repaired.append(item_top5_predictions)
        # -------------------------------------------------------------------- #
        # No loss, no completion prediction accuracies.
        mc_full_top1_accuracy[i_mc - MC_runs[0]] = np.sum(np.equal(top1_pred_full_model,true_labels))/len(true_labels)
        mc_cloud_top1_accuracy[i_mc - MC_runs[0]] = np.sum(np.equal(top1_pred_split_model,true_labels))/len(true_labels)
        score_full = 0
        score_split = 0
        for ks in range(len(true_labels)):
            if true_labels[ks] in top5_pred_full_model[ks]:
                score_full+=1
            if true_labels[ks] in top5_pred_split_model[ks]:
                score_split+=1
        mc_full_top5_accuracy[i_mc - MC_runs[0]] = score_full/len(true_labels)
        mc_cloud_top5_accuracy[i_mc - MC_runs[0]] = score_split/len(true_labels)
        # -------------------------------------------------------------------- #
        # error concealment accuracies.
        if ecDict != 'noEC':
            mc_repaired_top1_accuracy[i_mc - MC_runs[0]] = np.sum(np.equal(top1_pred_repaired,true_labels))/len(true_labels)
            score_repaired = 0
            for ks in range(len(true_labels)):
                if true_labels[ks] in top5_pred_repaired[ks]:
                    score_repaired+=1
            mc_repaired_top5_accuracy[i_mc - MC_runs[0]] = score_repaired/len(true_labels)
    # -----------------------------------------------------------------------= #
    print('Summary of experiment')
    print('The Top 1 accuracy score was')
    print(f'with full model {mc_full_top1_accuracy}')
    print(f'with no error concealment, CI Top 1 accuracy {mc_cloud_top1_accuracy}')
    print('The Top 5 accuracy score was')
    print(f'with full model {mc_full_top5_accuracy}')
    print(f'with no error concealment, CI Top 5 accuracy {mc_cloud_top5_accuracy}')
    if ecDict != 'noEC':
        print(f'with {ec_method}, Top 1 accuracy is {mc_repaired_top1_accuracy}')
        print(f'with {ec_method}, Top 5 accuracy is {mc_repaired_top5_accuracy}')

    res_filename = channel_flag+'_'+str(numberOfBits_1)+'Bits_'+str(numberOfBits_2)+'Bits_rpp_'+str(rowsPerPacket)+'_MC_'+str(MC_runs[0])+'_'+str(MC_runs[1])+'_'
    if ecDict != 'noEC':
        res_filename = res_filename+ec_method
        df = pd.DataFrame({'full_top1_accuracy':mc_full_top1_accuracy,'cloud_top1_accuracy':mc_cloud_top1_accuracy, \
        'full_top5_accuracy':mc_full_top5_accuracy,'cloud_top5_accuracy':mc_cloud_top5_accuracy, \
        'mc_repaired_top1_accuracy':mc_repaired_top1_accuracy,'mc_repaired_top5_accuracy':mc_repaired_top5_accuracy,'mc_time':mc_time})
        df.to_csv(os.path.join(results_dir,res_filename+'.csv'),index=False)

    else:
        df = pd.DataFrame({'full_top1_accuracy':mc_full_top1_accuracy,'cloud_top1_accuracy':mc_cloud_top1_accuracy, \
        'full_top5_accuracy':mc_full_top5_accuracy,'cloud_top5_accuracy':mc_cloud_top5_accuracy})
        df.to_csv(os.path.join(results_dir,res_filename+'.csv'),index=False)

# ---------------------------------------------------------------------------- #
