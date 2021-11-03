#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 15:18:08 2021.

@author: Ashiv Hans Dhondea
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['DejaVu Sans']})
rc('text',usetex=True)
params={'text.latex.preamble':[r'\usepackage{amsmath}',r'\usepackage{amssymb}']}
plt.rcParams.update(params);
from scipy import stats
# --------------------------------------------------------------------------- #
model_name = 'resnet18'#'resnet34'#'efficientnetb0'#'densenet121'#
model_name_str = 'ResNet-18'# 'ResNet-34'#'EfficientNet-B0'#'DenseNet-121'#'
splitLayer ='add_3'#'block2b_add'#'pool2_conv'#'add_1'# 
rowsPerPacket = 4
quant1 = 8
quant2 = 8
results_dir = 'simData'
dataset = 'largeTest'
output_dir = os.path.join('mc_results',dataset,model_name+'_'+splitLayer)

os.makedirs(output_dir,exist_ok=True)
file_str = 'GC_'+str(quant1)+'Bits_'+str(quant2)+'Bits_rpp_'+str(rowsPerPacket)+'_MC_'
# --------------------------------------------------------------------------- #
lossProbability = [0.3,0.2,0.1,0.01]
burstLength = [1,2,3,4,5,6,7] # [1,2,3,4,5,6,7,8,9,10,11,12,13,14] #

step_MC = 5
num_MC = 20
MC_index = [i for i in range(0,num_MC+step_MC,step_MC)]

cloud_top1_acc = np.zeros([len(lossProbability),len(burstLength),num_MC],dtype=np.float64)
cloud_top5_acc = np.zeros_like(cloud_top1_acc)
full_top1_acc = np.zeros_like(cloud_top1_acc)
full_top5_acc = np.zeros_like(cloud_top1_acc)

caltec_top1_acc = np.zeros_like(cloud_top1_acc)
caltec_top5_acc = np.zeros_like(caltec_top1_acc)
altec_top1_acc = np.zeros_like(caltec_top1_acc)
altec_top5_acc = np.zeros_like(altec_top1_acc)
halrtc_top1_acc = np.zeros_like(caltec_top1_acc)
halrtc_top5_acc = np.zeros_like(caltec_top5_acc)
silrtc_top1_acc = np.zeros_like(halrtc_top1_acc)
silrtc_top5_acc = np.zeros_like(silrtc_top1_acc)
ns_top1_acc = np.zeros_like(caltec_top1_acc)
ns_top5_acc = np.zeros_like(caltec_top5_acc)

cloud_top1_mean = np.zeros([len(lossProbability),len(burstLength)],dtype=np.float64)
cloud_top5_mean = np.zeros_like(cloud_top1_mean)
full_top1_mean = np.zeros_like(cloud_top1_mean)
full_top5_mean = np.zeros_like(cloud_top1_mean)

caltec_top1_mean = np.zeros_like(cloud_top1_mean)
caltec_top5_mean = np.zeros_like(cloud_top5_mean)
altec_top1_mean = np.zeros_like(cloud_top1_mean)
altec_top5_mean = np.zeros_like(cloud_top5_mean)
halrtc_top1_mean = np.zeros_like(caltec_top1_mean)
halrtc_top5_mean = np.zeros_like(caltec_top1_mean)
silrtc_top1_mean = np.zeros_like(caltec_top1_mean)
silrtc_top5_mean = np.zeros_like(caltec_top1_mean)
ns_top1_mean = np.zeros_like(caltec_top1_mean)
ns_top5_mean = np.zeros_like(caltec_top5_mean)
# --------------------------------------------------------------------------- #
for i_lp in range(len(lossProbability)):
    for i_bl in range(len(burstLength)):
        print(f'loss probability {lossProbability[i_lp]} burst length {burstLength[i_bl]}')
        df_results = pd.read_csv(os.path.join(results_dir,dataset,model_name,splitLayer+'_lp_'+str(lossProbability[i_lp])+'_Bl_'+str(burstLength[i_bl]),file_str+str(MC_index[0])+'_'+str(MC_index[-1])+'_.csv'))
        full_top1_acc[i_lp,i_bl,:] = df_results['full_top1_accuracy'].to_numpy()
        cloud_top1_acc[i_lp,i_bl,:] = df_results['cloud_top1_accuracy'].to_numpy()
        full_top5_acc[i_lp,i_bl,:] = df_results['full_top5_accuracy'].to_numpy()
        cloud_top5_acc[i_lp,i_bl,:] = df_results['cloud_top5_accuracy'].to_numpy()
        
        full_top1_mean[i_lp,i_bl] = np.mean(full_top1_acc[i_lp,i_bl,:])
        full_top5_mean[i_lp,i_bl] = np.mean(full_top5_acc[i_lp,i_bl,:])
        cloud_top1_mean[i_lp,i_bl] = np.mean(cloud_top1_acc[i_lp,i_bl,:])
        cloud_top5_mean[i_lp,i_bl] = np.mean(cloud_top5_acc[i_lp,i_bl,:])
        
        for i_mc in range(len(MC_index)-1):
            df_results = pd.read_csv(os.path.join(results_dir,dataset,model_name,splitLayer+'_lp_'+str(lossProbability[i_lp])+'_Bl_'+str(burstLength[i_bl]),file_str+str(MC_index[i_mc])+'_'+str(MC_index[i_mc+1])+'_CALTeC.csv'))
            caltec_top1_acc[i_lp,i_bl,MC_index[i_mc]:MC_index[i_mc+1]] = df_results['mc_repaired_top1_accuracy'].to_numpy()
            caltec_top5_acc[i_lp,i_bl,MC_index[i_mc]:MC_index[i_mc+1]] = df_results['mc_repaired_top5_accuracy'].to_numpy()
            
            df_results = pd.read_csv(os.path.join(results_dir,dataset,model_name,splitLayer+'_lp_'+str(lossProbability[i_lp])+'_Bl_'+str(burstLength[i_bl]),file_str+str(MC_index[i_mc])+'_'+str(MC_index[i_mc+1])+'_ALTeC.csv'))
            altec_top1_acc[i_lp,i_bl,MC_index[i_mc]:MC_index[i_mc+1]] = df_results['mc_repaired_top1_accuracy'].to_numpy()
            altec_top5_acc[i_lp,i_bl,MC_index[i_mc]:MC_index[i_mc+1]] = df_results['mc_repaired_top5_accuracy'].to_numpy()
            
            df_results = pd.read_csv(os.path.join(results_dir,dataset,model_name,splitLayer+'_lp_'+str(lossProbability[i_lp])+'_Bl_'+str(burstLength[i_bl]),file_str+str(MC_index[i_mc])+'_'+str(MC_index[i_mc+1])+'_HaLRTC.csv'))
            halrtc_top1_acc[i_lp,i_bl,MC_index[i_mc]:MC_index[i_mc+1]] = df_results['mc_repaired_top1_accuracy'].to_numpy()
            halrtc_top5_acc[i_lp,i_bl,MC_index[i_mc]:MC_index[i_mc+1]] = df_results['mc_repaired_top5_accuracy'].to_numpy()
            
            df_results = pd.read_csv(os.path.join(results_dir,dataset,model_name,splitLayer+'_lp_'+str(lossProbability[i_lp])+'_Bl_'+str(burstLength[i_bl]),file_str+str(MC_index[i_mc])+'_'+str(MC_index[i_mc+1])+'_SiLRTC.csv'))
            silrtc_top1_acc[i_lp,i_bl,MC_index[i_mc]:MC_index[i_mc+1]] = df_results['mc_repaired_top1_accuracy'].to_numpy()
            silrtc_top5_acc[i_lp,i_bl,MC_index[i_mc]:MC_index[i_mc+1]] = df_results['mc_repaired_top5_accuracy'].to_numpy()
            
            df_results = pd.read_csv(os.path.join(results_dir,dataset,model_name,splitLayer+'_lp_'+str(lossProbability[i_lp])+'_Bl_'+str(burstLength[i_bl]),file_str+str(MC_index[i_mc])+'_'+str(MC_index[i_mc+1])+'_InpaintNS.csv'))
            ns_top1_acc[i_lp,i_bl,MC_index[i_mc]:MC_index[i_mc+1]] = df_results['mc_repaired_top1_accuracy'].to_numpy()
            ns_top5_acc[i_lp,i_bl,MC_index[i_mc]:MC_index[i_mc+1]] = df_results['mc_repaired_top5_accuracy'].to_numpy()


        caltec_top1_mean[i_lp,i_bl] = np.mean(caltec_top1_acc[i_lp,i_bl,:])
        caltec_top5_mean[i_lp,i_bl] = np.mean(caltec_top5_acc[i_lp,i_bl,:])
        altec_top1_mean[i_lp,i_bl] = np.mean(altec_top1_acc[i_lp,i_bl,:])
        altec_top5_mean[i_lp,i_bl] = np.mean(altec_top5_acc[i_lp,i_bl,:])
        halrtc_top1_mean[i_lp,i_bl] = np.mean(halrtc_top1_acc[i_lp,i_bl,:])
        halrtc_top5_mean[i_lp,i_bl] = np.mean(halrtc_top5_acc[i_lp,i_bl,:])
        silrtc_top1_mean[i_lp,i_bl] = np.mean(silrtc_top1_acc[i_lp,i_bl,:])
        silrtc_top5_mean[i_lp,i_bl] = np.mean(silrtc_top5_acc[i_lp,i_bl,:])
        ns_top1_mean[i_lp,i_bl] = np.mean(ns_top1_acc[i_lp,i_bl,:])
        ns_top5_mean[i_lp,i_bl] = np.mean(ns_top5_acc[i_lp,i_bl,:])
        
# --------------------------------------------------------------------------- #
color_list = ['limegreen','crimson','dimgray','darkorange','mediumblue','magenta','cyan','darkviolet','saddlebrown','maroon']
marker_list = ['^','2','d','>','.','+','s','x','<','o','v']

chosen_fontsize = 12

for i_lp in range(len(lossProbability)):
    metric_str = 'Top1'
    fig = plt.figure(i_lp);
    ax = fig.gca();
    plt.rc('text',usetex=True)
    params= {'legend.fontsize':chosen_fontsize,'legend.handlelength':2}
    plt.rcParams.update(params)
    
    plt.title(r'\textbf{Monte Carlo experiments with} \verb|%s| \verb|%s| \textbf{tensors with} $P_B = %s$' %(model_name_str,splitLayer,str(lossProbability[i_lp])),fontsize=chosen_fontsize)
    plt.ylim(-0.05,1.05)
    plt.plot(burstLength,full_top1_mean[i_lp,:],marker=marker_list[0],color=color_list[0],label=r'Top-1 w/ No Loss')
    plt.plot(burstLength,cloud_top1_mean[i_lp,:],marker=marker_list[1],color=color_list[1],label=r'Top-1 w/ No Completion')
    plt.plot(burstLength,caltec_top1_mean[i_lp,:],marker=marker_list[2],color=color_list[2],label=r'Top-1 w/ CALTeC')
    plt.plot(burstLength,altec_top1_mean[i_lp,:],marker=marker_list[3],color=color_list[3],label=r'Top-1 w/ ALTeC')
    plt.plot(burstLength,halrtc_top1_mean[i_lp,:],marker=marker_list[4],color=color_list[4],label=r'Top-1 w/ HaLRTC')
    plt.plot(burstLength,silrtc_top1_mean[i_lp,:],marker=marker_list[5],color=color_list[5],label=r'Top-1 w/ SiLRTC')
    plt.plot(burstLength,ns_top1_mean[i_lp,:],marker=marker_list[6],color=color_list[6],label=r'Top-1 w/ Navier-Stokes')
    
    
    plt.xticks(burstLength,fontsize=chosen_fontsize)
    ax.set_xlabel(r'$L_B$',fontsize=chosen_fontsize)
    ax.set_ylabel(r'\textbf{Accuracy}',fontsize=chosen_fontsize)
    plt.yticks([0.,0.3,0.5,0.7,0.9],fontsize=chosen_fontsize)
    plt.grid()
    
    plt.legend(loc='best')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir,'rpp_'+str(rowsPerPacket)+'_lp_'+str(lossProbability[i_lp])+'_'+metric_str+'.pdf'))
    fig.clf()
    plt.close()
    

for i_lp in range(len(lossProbability)):
    metric_str = 'Top5'
    fig = plt.figure(i_lp);
    ax = fig.gca();
    plt.rc('text',usetex=True)
    params= {'legend.fontsize':chosen_fontsize,'legend.handlelength':2}
    plt.rcParams.update(params)
    
    plt.title(r'\textbf{Monte Carlo experiments with} \verb|%s| \verb|%s| \textbf{tensors with} $P_B = %s$' %(model_name_str,splitLayer,str(lossProbability[i_lp])),fontsize=chosen_fontsize)
    plt.ylim(-0.05,1.05)
    plt.plot(burstLength,full_top5_mean[i_lp,:],linestyle='dashed',marker=marker_list[0],color=color_list[0],label=r'Top-5 w/ No Loss')
    plt.plot(burstLength,cloud_top5_mean[i_lp,:],linestyle='dashed',marker=marker_list[1],color=color_list[1],label=r'Top-5 w/ No Completion')
    plt.plot(burstLength,caltec_top5_mean[i_lp,:],linestyle='dashed',marker=marker_list[2],color=color_list[2],label=r'Top-5 w/ CALTeC')
    plt.plot(burstLength,altec_top5_mean[i_lp,:],linestyle='dashed',marker=marker_list[3],color=color_list[3],label=r'Top-5 w/ ALTeC')
    plt.plot(burstLength,halrtc_top5_mean[i_lp,:],linestyle='dashed',marker=marker_list[4],color=color_list[4],label=r'Top-5 w/ HaLRTC')
    plt.plot(burstLength,silrtc_top5_mean[i_lp,:],linestyle='dashed',marker=marker_list[5],color=color_list[5],label=r'Top-5 w/ SiLRTC')
    plt.plot(burstLength,ns_top5_mean[i_lp,:],linestyle='dashed',marker=marker_list[6],color=color_list[6],label=r'Top-5 w/ Navier-Stokes')
    
    plt.xticks(burstLength,fontsize=chosen_fontsize)
    ax.set_xlabel(r'$L_B$',fontsize=chosen_fontsize)
    ax.set_ylabel(r'\textbf{Accuracy}',fontsize=chosen_fontsize)
    plt.yticks([0.,0.3,0.5,0.7,0.9],fontsize=chosen_fontsize)
    plt.grid()
    
    plt.legend(loc='best')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir,'rpp_'+str(rowsPerPacket)+'_lp_'+str(lossProbability[i_lp])+'_'+metric_str+'.pdf'))
    fig.clf()
    plt.close()    
# --------------------------------------------------------------------------- #
for i_lp in range(len(lossProbability)):
    bl_top1_mean = np.mean(full_top1_acc[i_lp,:,:],axis=1)
    bl_top1_std = np.std(full_top1_acc[i_lp,:,:],axis=1)
    bl_top5_mean = np.mean(full_top5_acc[i_lp,:,:],axis=1)
    bl_top5_std = np.std(full_top5_acc[i_lp,:,:],axis=1)
    bl_nc_top1_mean = np.mean(cloud_top1_acc[i_lp,:,:],axis=1)
    bl_nc_top1_std = np.std(cloud_top1_acc[i_lp,:,:],axis=1)
    bl_nc_top5_mean = np.mean(cloud_top5_acc[i_lp,:,:],axis=1)
    bl_nc_top5_std = np.std(cloud_top5_acc[i_lp,:,:],axis=1)
    bl_caltec_top1_mean = np.mean(caltec_top1_acc[i_lp,:,:],axis=1)
    bl_caltec_top1_std = np.std(caltec_top1_acc[i_lp,:,:],axis=1)
    bl_caltec_top5_mean = np.mean(caltec_top5_acc[i_lp,:,:],axis=1)
    bl_caltec_top5_std = np.std(caltec_top5_acc[i_lp,:,:],axis=1)
    bl_altec_top1_mean = np.mean(altec_top1_acc[i_lp,:,:],axis=1)
    bl_altec_top1_std = np.std(altec_top1_acc[i_lp,:,:],axis=1)
    bl_altec_top5_mean = np.mean(altec_top5_acc[i_lp,:,:],axis=1)
    bl_altec_top5_std = np.std(altec_top5_acc[i_lp,:,:],axis=1)
    bl_halrtc_top1_mean = np.mean(halrtc_top1_acc[i_lp,:,:],axis=1)
    bl_halrtc_top1_std = np.std(halrtc_top1_acc[i_lp,:,:],axis=1)
    bl_halrtc_top5_mean = np.mean(halrtc_top5_acc[i_lp,:,:],axis=1)
    bl_halrtc_top5_std = np.std(halrtc_top5_acc[i_lp,:,:],axis=1)
    bl_silrtc_top1_mean = np.mean(silrtc_top1_acc[i_lp,:,:],axis=1)
    bl_silrtc_top1_std = np.std(silrtc_top1_acc[i_lp,:,:],axis=1)
    bl_silrtc_top5_mean = np.mean(silrtc_top5_acc[i_lp,:,:],axis=1)
    bl_silrtc_top5_std = np.std(silrtc_top5_acc[i_lp,:,:],axis=1)
    bl_ns_top1_mean = np.mean(ns_top1_acc[i_lp,:,:],axis=1)
    bl_ns_top1_std = np.std(ns_top1_acc[i_lp,:,:],axis=1)
    bl_ns_top5_mean = np.mean(ns_top5_acc[i_lp,:,:],axis=1)
    bl_ns_top5_std = np.std(ns_top5_acc[i_lp,:,:],axis=1)
    
    df_lp = pd.DataFrame({'NL_Top1_mean':bl_top1_mean,'NL_Top1_std':bl_top1_std,'NL_Top5_mean':bl_top5_mean,'NL_Top5_std':bl_top5_std,
                          'NC_Top1_mean':bl_nc_top1_mean,'NC_Top1_std':bl_nc_top1_std,'NC_Top5_mean':bl_nc_top5_mean,'NC_Top5_std':bl_nc_top5_std,
                          'CALTeC_Top1_mean':bl_caltec_top1_mean,'CALTeC_Top1_std':bl_caltec_top1_std,'CALTeC_Top5_mean':bl_caltec_top5_mean,'CALTeC_Top5_std':bl_caltec_top5_std,
                          'ALTeC_Top1_mean':bl_altec_top1_mean,'ALTeC_Top1_std':bl_altec_top1_std,'ALTeC_Top5_mean':bl_altec_top5_mean,'ALTeC_Top5_std':bl_altec_top5_std,
                          'HaLRTC_Top1_mean':bl_halrtc_top1_mean,'HaLRTC_Top1_std':bl_halrtc_top1_std,'HaLRTC_Top5_mean':bl_halrtc_top5_mean,'HaLRTC_Top5_std':bl_halrtc_top5_std,
                          'SiLRTC_Top1_mean':bl_silrtc_top1_mean,'SilRTC_Top1_std':bl_silrtc_top1_std,'SiLRTC_Top5_mean':bl_silrtc_top5_mean,'SiLRTC_Top5_std':bl_silrtc_top5_std,
                          'NS_Top1_mean':bl_ns_top1_mean,'NS_Top1_std':bl_ns_top1_std,'NS_Top5_mean':bl_ns_top5_mean,'NS_Top5_std':bl_ns_top5_std
                          })
    df_lp.to_csv(os.path.join(output_dir,'ImgClass_MonteCarlo_rpp_'+str(rowsPerPacket)+'_lp_'+str(lossProbability[i_lp])+'.csv'))
    # df_lp.to_latex(os.path.join(output_dir,'ImgClass_MonteCarlo'+'rpp_'+str(rowsPerPacket)+'_lp_'+str(lossProbability[i_lp])+'.tex'))


print('Calculating stats')     
full_top1_acc_lp = np.reshape(full_top1_acc,(len(lossProbability),len(burstLength)*(num_MC))) 
cloud_top1_acc_lp = np.reshape(cloud_top1_acc,(len(lossProbability),len(burstLength)*(num_MC)))
caltec_top1_lp = np.reshape(caltec_top1_acc,(len(lossProbability),len(burstLength)*(num_MC)))
altec_top1_lp = np.reshape(altec_top1_acc,(len(lossProbability),len(burstLength)*(num_MC)))
halrtc_top1_lp = np.reshape(halrtc_top1_acc,(len(lossProbability),len(burstLength)*(num_MC)))
silrtc_top1_lp = np.reshape(silrtc_top1_acc,(len(lossProbability),len(burstLength)*(num_MC)))
ns_top1_lp = np.reshape(ns_top1_acc,(len(lossProbability),len(burstLength)*(num_MC)))

full_top5_acc_lp = np.reshape(full_top5_acc,(len(lossProbability),len(burstLength)*(num_MC))) 
cloud_top5_acc_lp = np.reshape(cloud_top5_acc,(len(lossProbability),len(burstLength)*(num_MC)))
caltec_top5_lp = np.reshape(caltec_top5_acc,(len(lossProbability),len(burstLength)*(num_MC)))
altec_top5_lp = np.reshape(altec_top5_acc,(len(lossProbability),len(burstLength)*(num_MC)))
halrtc_top5_lp = np.reshape(halrtc_top5_acc,(len(lossProbability),len(burstLength)*(num_MC)))
silrtc_top5_lp = np.reshape(silrtc_top5_acc,(len(lossProbability),len(burstLength)*(num_MC)))
ns_top5_lp = np.reshape(ns_top5_acc,(len(lossProbability),len(burstLength)*(num_MC)))
        
t_caltec_mean_top1 = np.mean(caltec_top1_lp,axis=1)
t_caltec_std_top1 = np.std(caltec_top1_lp)

t_altec_mean_top1 = np.mean(altec_top1_lp,axis=1)
t_altec_std_top1 = np.std(altec_top1_lp)

t_halrtc_mean_top1 = np.mean(halrtc_top1_lp,axis=1)
t_halrtc_std_top1 = np.std(halrtc_top1_lp,axis=1)

t_silrtc_mean_top1 = np.mean(silrtc_top1_lp,axis=1)
t_silrtc_std_top1 = np.std(silrtc_top1_lp,axis=1)

t_ns_mean_top1 = np.mean(ns_top1_lp,axis=1)
t_ns_std_top1 = np.std(ns_top1_lp,axis=1)

t_caltec_mean_top5 = np.mean(caltec_top5_lp,axis=1)
t_caltec_std_top5 = np.std(caltec_top5_lp)

t_altec_mean_top5 = np.mean(altec_top5_lp,axis=1)
t_altec_std_top5 = np.std(altec_top5_lp)

t_halrtc_mean_top5 = np.mean(halrtc_top5_lp,axis=1)
t_halrtc_std_top5 = np.std(halrtc_top5_lp,axis=1)

t_silrtc_mean_top5 = np.mean(silrtc_top5_lp,axis=1)
t_silrtc_std_top5 = np.std(silrtc_top5_lp,axis=1)

t_ns_mean_top5 = np.mean(ns_top5_lp,axis=1)
t_ns_std_top5 = np.std(ns_top5_lp,axis=1)

t_caltec_altec_top1 = np.zeros([len(lossProbability)],dtype=np.float64)
g_caltec_altec_top1 = np.zeros_like(t_caltec_altec_top1)

t_caltec_halrtc_top1 = np.zeros([len(lossProbability)],dtype=np.float64)
g_caltec_halrtc_top1 = np.zeros_like(t_caltec_halrtc_top1)

t_caltec_silrtc_top1 = np.zeros([len(lossProbability)],dtype=np.float64)
g_caltec_silrtc_top1 = np.zeros_like(t_caltec_silrtc_top1)

t_caltec_ns_top1 = np.zeros([len(lossProbability)],dtype=np.float64)
g_caltec_ns_top1 = np.zeros([len(lossProbability)],dtype=np.float64)

t_nc_mean_top1 = np.mean(cloud_top1_acc_lp,axis=1)
t_nc_std_top1 = np.std(cloud_top1_acc_lp,axis=1)

t_nc_mean_top5 = np.mean(cloud_top5_acc_lp,axis=1)
t_nc_std_top5 = np.std(cloud_top5_acc_lp,axis=1)

t_nl_mean_top1 = np.mean(full_top1_acc_lp,axis=1)
t_nl_std_top1 = np.std(full_top1_acc_lp,axis=1)

t_nl_mean_top5 = np.mean(full_top5_acc_lp,axis=1)
t_nl_std_top5 = np.std(full_top5_acc_lp,axis=1)


for i_lp in range(len(lossProbability)):
    t_caltec_altec_top1[i_lp], g_caltec_altec_top1[i_lp] = stats.ttest_ind(caltec_top1_lp[i_lp,:],altec_top1_lp[i_lp,:],equal_var=False)
    t_caltec_halrtc_top1[i_lp],g_caltec_halrtc_top1[i_lp] = stats.ttest_ind(caltec_top1_lp[i_lp,:],halrtc_top1_lp[i_lp,:],equal_var=False)
    t_caltec_silrtc_top1[i_lp],g_caltec_silrtc_top1[i_lp] = stats.ttest_ind(caltec_top1_lp[i_lp,:],silrtc_top1_lp[i_lp,:],equal_var=False)
    t_caltec_ns_top1[i_lp],g_caltec_ns_top1[i_lp] = stats.ttest_ind(caltec_top1_lp[i_lp,:],ns_top1_lp[i_lp,:],equal_var=False)


df_lp = pd.DataFrame({'caltec_altec':g_caltec_altec_top1,'caltec_halrtc':g_caltec_halrtc_top1,'caltec_silrtc':g_caltec_silrtc_top1,'caltec_ns':g_caltec_ns_top1})
df_lp.to_latex(os.path.join(output_dir,'rpp_'+str(rowsPerPacket)+'_ttest_top1.tex'))


fig = plt.figure(21);
ax = fig.gca();
params= {'legend.fontsize':chosen_fontsize,'legend.handlelength':2}
plt.rcParams.update(params)

#plt.title(r'\textbf{Default settings experiments with} \verb|%s| \verb|%s| \textbf{tensors.}' %(model_name_str,splitLayer),fontsize=chosen_fontsize)
plt.ylim(-0.08,1.08)

plt.plot(lossProbability,t_nl_mean_top1,linewidth=2,color= color_list[3],label=r'Top-1 w/ No Loss')
plt.plot(lossProbability,t_halrtc_mean_top1,linestyle='dashed',linewidth=2,marker=marker_list[4],markersize=12,color = color_list[4],label=r'Top-1 w/ HaLRTC')
plt.plot(lossProbability,t_caltec_mean_top1,linewidth=2,marker=marker_list[1],markersize=12,color= color_list[1],label=r'Top-1 w/ CALTeC')
plt.plot(lossProbability,t_ns_mean_top1,linestyle='dashdot',linewidth=2,marker=marker_list[7],markersize=12,color=color_list[7],label=r'Top-1 w/ Navier-Stokes')
plt.plot(lossProbability,t_altec_mean_top1,linestyle='dotted',linewidth=2,marker=marker_list[0],markersize=12,color = color_list[0],label=r'Top-1 w/ ALTeC')
plt.plot(lossProbability,t_silrtc_mean_top1,linewidth=2,marker=marker_list[5],markersize=12,color= color_list[5],label=r'Top-1 w/ SiLRTC')
plt.plot(lossProbability,t_nc_mean_top1,linewidth=2,marker=marker_list[2],markersize=12,color= color_list[2],label=r'Top-1 w/ No Completion')

plt.xticks(lossProbability,fontsize=chosen_fontsize)
ax.set_xlabel(r'$P_B$',fontsize=chosen_fontsize)
ax.set_ylabel(r'\textbf{Top-1 Accuracy}',fontsize=chosen_fontsize)
plt.yticks([0.1,0.3,0.5,0.7,0.9],fontsize=chosen_fontsize)
plt.grid()

# plt.tight_layout()
# fig.savefig(os.path.join(output_dir,'lp_accumulated_vcip_rpp_'+str(rowsPerPacket)+'_top1.pdf'))
# fig.clf()
# plt.close()
lgd = ax.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
fig.savefig(os.path.join(output_dir,'lp_accumulated_vcip_rpp_'+str(rowsPerPacket)+'_top1_legend.pdf'), bbox_extra_artists=(lgd,), bbox_inches='tight')


fig = plt.figure(22);
ax = fig.gca();
params= {'legend.fontsize':chosen_fontsize,'legend.handlelength':2}
plt.rcParams.update(params)

#plt.title(r'\textbf{Default settings experiments with} \verb|%s| \verb|%s| \textbf{tensors.}' %(model_name_str,splitLayer),fontsize=chosen_fontsize)
plt.ylim(-0.08,1.08)

plt.plot(lossProbability,t_nl_mean_top5,linewidth=2,color= color_list[3],label=r'Top-5 w/ No Loss')
plt.plot(lossProbability,t_halrtc_mean_top5,linestyle='dashed',linewidth=2,marker=marker_list[4],markersize=12,color = color_list[4],label=r'Top-5 w/ HaLRTC')
plt.plot(lossProbability,t_caltec_mean_top5,linewidth=2,marker=marker_list[1],markersize=12,color= color_list[1],label=r'Top-5 w/ CALTeC')
plt.plot(lossProbability,t_ns_mean_top5,linestyle='dashdot',linewidth=2,marker=marker_list[7],markersize=12,color=color_list[7],label=r'Top-5 w/ Navier-Stokes')
plt.plot(lossProbability,t_altec_mean_top5,linestyle='dotted',linewidth=2,marker=marker_list[0],markersize=12,color = color_list[0],label=r'Top-5 w/ ALTeC')
plt.plot(lossProbability,t_silrtc_mean_top5,linewidth=2,marker=marker_list[5],markersize=12,color= color_list[5],label=r'Top-5 w/ SiLRTC')
plt.plot(lossProbability,t_nc_mean_top5,linewidth=2,marker=marker_list[2],markersize=12,color= color_list[2],label=r'Top-5 w/ No Completion')


plt.xticks(lossProbability,fontsize=chosen_fontsize)
ax.set_xlabel(r'$P_B$',fontsize=chosen_fontsize)
ax.set_ylabel(r'\textbf{Top-5 Accuracy}',fontsize=chosen_fontsize)
plt.yticks([0.1,0.3,0.5,0.7,0.9],fontsize=chosen_fontsize)
plt.grid()

# plt.tight_layout()
# fig.savefig(os.path.join(output_dir,'lp_accumulated_vcip_rpp_'+str(rowsPerPacket)+'_top5.pdf'))
# fig.clf()
# plt.close()

lgd = ax.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
fig.savefig(os.path.join(output_dir,'lp_accumulated_vcip_rpp_'+str(rowsPerPacket)+'_top5_legend.pdf'), bbox_extra_artists=(lgd,), bbox_inches='tight')
