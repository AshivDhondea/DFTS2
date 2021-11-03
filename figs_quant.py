#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 15:16:22 2021.

Process quantization experiment results.

@author: Ashiv Hans Dhondea
"""
import sys,os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['DejaVu Sans']})
rc('text',usetex=True)
params={'text.latex.preamble':[r'\usepackage{amsmath}',r'\usepackage{amssymb}']}
plt.rcParams.update(params);
# --------------------------------------------------------------------------- #
# boilerplate code to get the script's file name
script_name = os.path.basename(sys.argv[0])[:-3];
# ---------------------------------------------------------------------------- #
# model name str for printing on figures.
model_name_str ='ResNet-18'#'ResNet-34'# 'DenseNet-121'#'EfficientNet-B0'#  
loaded_model_name ='resnet18'#'resnet34'#'densenet121'# 'efficientnetb0'# 
splitLayer ='add_1' # 'pool2_conv'# 'block2b_add'#
test_set = 'largeTest'
rowsPerPacket = 8
main_dir = os.path.join('simData',test_set,loaded_model_name,splitLayer+'_NoChannel')

quant_res_bits_1 = [2,4,6,8,10]

full_top1 = np.zeros([len(quant_res_bits_1)],dtype=np.float64)
full_top5 = np.zeros([len(quant_res_bits_1)],dtype=np.float64)
quant_top1 = np.zeros_like(full_top1)
quant_top5 = np.zeros_like(full_top5)

for i_q in range(len(quant_res_bits_1)):
    results_path = os.path.join(main_dir,'NC_'+str(quant_res_bits_1[i_q])+'Bits_8Bits_rpp_'+str(rowsPerPacket)+'_MC_0_1_.csv')
    df = pd.read_csv(results_path)
    full_top1[i_q] = df['full_top1_accuracy'].to_numpy()
    full_top5[i_q] = df['full_top5_accuracy'].to_numpy()
    quant_top1[i_q] = df['cloud_top1_accuracy'].to_numpy()
    quant_top5[i_q] = df['cloud_top5_accuracy'].to_numpy()
    
output_dir = os.path.join('output_figs',script_name)
os.makedirs(output_dir,exist_ok=True)

# --------------------------------------------------------------------------- #
color_list = ['limegreen','crimson','dimgray','darkorange','mediumblue','magenta','cyan','darkviolet','saddlebrown','maroon']
marker_list = ['^','2','d','>','.','+','s','x','<','o','v']
chosen_fontsize = 12

fig = plt.figure(1);
ax = fig.gca();
plt.rc('text',usetex=True)
plt.rc('font',family='serif');
plt.rc('font',family='DejaVu Sans');
params= {'legend.fontsize':chosen_fontsize,'legend.handlelength':2}
plt.rcParams.update(params)

plt.title(r'\textbf{Classification accuracy with} \verb|%s| \verb|%s| \textbf{quantized tensors.}' %(model_name_str,splitLayer),fontsize=chosen_fontsize)
plt.ylim(0.0,1.05)
plt.plot(quant_res_bits_1,full_top1,color=color_list[0],marker=marker_list[0],label=r'Top-1 No quantization')
plt.plot(quant_res_bits_1,full_top5,color_list[0],marker=marker_list[0],linestyle='dashed',label=r'Top-5 No quantization')

plt.plot(quant_res_bits_1,quant_top1,color_list[1],marker=marker_list[1],label=r'Top-1 w/ quantization')
plt.plot(quant_res_bits_1,quant_top5,color_list[1],marker=marker_list[1],linestyle='dashed',label=r'Top-5 w/ quantization')


ax.set_xlabel(r'$n$\textbf{-bit quantization}')
ax.set_ylabel(r'\textbf{Accuracy}')
plt.grid()
plt.legend(loc='best')
plt.tight_layout()
fig.savefig(os.path.join(output_dir,'quant_'+loaded_model_name+'_'+splitLayer+'.pdf'))

