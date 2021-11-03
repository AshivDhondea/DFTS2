#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  20 18:40:10 2021.

@author: Ashiv Hans Dhondea
"""
import os
import numpy as np
import pandas as pd
# --------------------------------------------------------------------------- #
deep_models = ['efficientnetb0']#,'resnet18']#,'resnet18','resnet34','densenet121']
splitLayers = ['block2b_add']#'add_1']#,'add_3','add_3','pool2_conv']
rpps = [8]#,4,8,8,8] # rowsPerPacket
dataset = 'largeTest'
tensor_id = 0

lossProbability = [0.3]
burstLength = [1]
num_mc = 20
num_batches = 4
rowsPerPacket = 8

"""
lossMap is num_images x num_pkts_per_channel x num_channels.
"""


for i_m in range(len(deep_models)):
    for i_lp in range(len(lossProbability)):
        for i_bl in range(len(burstLength)):
            print(f'Processing {deep_models[i_m]} {splitLayers[i_m]} tensors for Pb {lossProbability[i_lp]} Lb {burstLength[i_bl]}')
            input_dir = os.path.join('lossMaps',dataset,deep_models[i_m],splitLayers[i_m]+'_lp_'+str(lossProbability[i_lp])+'_Bl_'+str(burstLength[i_bl]))
            output_dir = os.path.join('external_traces',dataset,deep_models[i_m],splitLayers[i_m])
                                     
            os.makedirs(output_dir,exist_ok=True)

            for i_mc in range(num_mc):
                lossMaps_acc = []
                for i_b in range(num_batches):
                     lossMap = np.load(os.path.join(input_dir,'MC_'+str(i_mc)+'_batch_'+str(i_b)+'_tensor_'+str(tensor_id)+'_rpp_'+str(rpps[i_m])+'.npy'))
                     lossMaps_acc.append(lossMap)
                lossMaps = np.concatenate(lossMaps_acc,axis=0)
                
                writer = pd.ExcelWriter(os.path.join(output_dir,'Rpp_'+str(rpps[i_m])+'_MC_'+str(i_mc)+'.xlsx'), engine='xlsxwriter')
                for i_c in range(np.shape(lossMaps)[-1]):
                    df = pd.DataFrame.from_records(lossMaps[:,:,i_c])
                    df.to_excel(writer, sheet_name=str(i_c))
                    df = df[0:0]
                writer.save()