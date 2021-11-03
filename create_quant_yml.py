#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 09:22:35 2021.

@author: Ashiv Hans Dhondea
"""
import yaml
import os

deep_model_name = 'efficientnetb0'
normalized = False
reshapeDims = [224,224]
splitLayer = 'block2b_add'
rowsPerPacket = 4

quant_res = [2,4,6,8,10]

results_dir = os.path.join('quant_yml',deep_model_name)
os.makedirs(results_dir,exist_ok=True)

filename = os.path.join('quant_yml','quant_template.yml')
with open(filename) as c:
        yml_dict = yaml.load(c,yaml.SafeLoader)
        
yml_dict['DeepModel']['fullModel'] = 'deep_models_full/'+deep_model_name+'_model.h5'
yml_dict['DeepModel']['normalize'] = normalized
yml_dict['DeepModel']['reshapeDims'] = reshapeDims
yml_dict['SplitLayer']['MobileModel'] = 'deep_models_split/'+deep_model_name+'_'+splitLayer+'_mobile_model.h5'
yml_dict['SplitLayer']['CloudModel'] = 'deep_models_split/'+deep_model_name+'_'+splitLayer+'_cloud_model.h5'
yml_dict['SplitLayer']['split'] = splitLayer
yml_dict['Transmission']['rowsperpacket'] = rowsPerPacket

yml_dict['Transmission']['quantization']['include'] = True


for i_q in range(len(quant_res)):
    yml_dict['Transmission']['quantization'][1]['numberOfBits'] = quant_res[i_q]
    
    fname_str = splitLayer+'_rpp_'+str(rowsPerPacket)+'_quant_'+str(quant_res[i_q])+'Bits.yml'
    f = open(os.path.join(results_dir,fname_str), "w")
    yaml.safe_dump(yml_dict,f,default_flow_style=None)
    f.close()
