#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 09:22:35 2021.

@author: Ashiv Hans Dhondea
"""
import yaml
import os

deep_model_name = 'resnet18'#'resnet34'#'densenet121'#'efficientnetb0'
normalized = 'False'#True
reshapeDims = [224,224]
splitLayer = 'add_3'#'pool2_conv'#'block2b_add'
quant1 = 8
rowsPerPacket = 4
MC_task = 'LoadLossPatterns'

lossProbability = [0.01,0.1,0.2,0.3]
burstLength = [1,2,3,4,5,6,7]#,8,9,10,11,12,13,14]

results_dir = os.path.join('caltec_yml',deep_model_name,splitLayer)
os.makedirs(results_dir,exist_ok=True)

filename = os.path.join('caltec_yml','caltec_template.yml')
with open(filename) as c:
        yml_dict = yaml.load(c,yaml.SafeLoader)
        
yml_dict['DeepModel']['fullModel'] = 'deep_models_full/'+deep_model_name+'_model.h5'
yml_dict['DeepModel']['normalize'] = normalized
yml_dict['DeepModel']['reshapeDims'] = reshapeDims
yml_dict['SplitLayer']['MobileModel'] = 'deep_models_split/'+deep_model_name+'_'+splitLayer+'_mobile_model.h5'
yml_dict['SplitLayer']['CloudModel'] = 'deep_models_split/'+deep_model_name+'_'+splitLayer+'_cloud_model.h5'
yml_dict['SplitLayer']['split'] = splitLayer
yml_dict['SimulationMode']['MonteCarlo']['include'] = True
yml_dict['SimulationMode']['MonteCarlo']['MC_task'] = MC_task
yml_dict['SimulationMode']['Demo']['include'] = False
yml_dict['Transmission']['rowsperpacket'] = rowsPerPacket
yml_dict['Transmission']['channel']['GilbertChannel']['include'] = True
yml_dict['Transmission']['quantization']['include'] = True
yml_dict['Transmission']['quantization'][1]['numberOfBits'] = quant1

yml_dict['ErrorConcealment']['CALTeC']['include'] = True
yml_dict['ErrorConcealment']['ALTeC']['include'] = False
yml_dict['ErrorConcealment']['HaLRTC']['include'] = False
yml_dict['ErrorConcealment']['HaLRTC']['numiters'] = 50
yml_dict['ErrorConcealment']['SiLRTC']['include'] = False
yml_dict['ErrorConcealment']['SiLRTC']['numiters'] = 50
yml_dict['ErrorConcealment']['InpaintNS']['include'] = False
yml_dict['ErrorConcealment']['InpaintNS']['radius'] = 5

MC_span = [0,5,10,15]

for i_lp in range(len(lossProbability)):
    for i_bl in range(len(burstLength)):
        for i_mc in range(len(MC_span)):
            yml_dict['SimulationMode']['MonteCarlo']['MC_runs'] = [MC_span[i_mc],MC_span[i_mc]+5]
            yml_dict['Transmission']['channel']['GilbertChannel']['lossProbability'] = lossProbability[i_lp]
            yml_dict['Transmission']['channel']['GilbertChannel']['burstLength'] = burstLength[i_bl]
            
            fname_str = 'lp_'+str(lossProbability[i_lp])+'_Bl_'+str(burstLength[i_bl])+'_rpp_'+str(rowsPerPacket)+'_'+str(quant1)+'Bits_MC_'+str(MC_span[i_mc])+'_'+str(MC_span[i_mc]+5)+'.yml'
            
            f = open(os.path.join(results_dir,fname_str), "w")
            yaml.safe_dump(yml_dict,f,default_flow_style=None)
            f.close()
