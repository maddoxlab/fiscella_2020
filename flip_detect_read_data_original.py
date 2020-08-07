#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:52:07 2019

@author: sarafiscella
"""


#import sys #fixed error of expyfun module
import numpy as np
import matplotlib.pyplot as plt #contains imread
from expyfun import (ExperimentController, fetch_data_file, analyze as ea,
                     building_doc)
#import random
from expyfun.visual import (Text, RawImage)
from expyfun.stimuli import (play_sound, rms, window_edges)
#import cv2
from expyfun.io import (read_wav, read_tab)
#from cv2 import VideoCapture, imread
#import pyglet
#from pyglet.image import AbstractImage
import pandas as pd
from pandas import DataFrame
from ast import literal_eval
from expyfun.analyze import dprime
import scipy
from scipy import stats
from matplotlib import rcParams
path = '/home/maddy/Code/data_analysis/Flip_Detect/Flip_Detect_Data/'
#elminate 012 because 50% at inverted
subjects = ['007','008','009','010','011','013','015','016','017','018','019','021','022','023','024','026','028', '029', '030', '031', '032', '034', '035']
rcParams['font.sans-serif'] = "Arial"
plt.rc('font', size=10)
plt.rc('axes', titlesize=10)


#arrays for graphs 
disagree_upright = np.zeros((len(subjects),2)) #upright video performance
disagree_inverted = np.zeros((len(subjects),2)) #inverted video performance
disagree_tvm = np.zeros((len(subjects),2)) #matching target vs masker (upright, inverted)
disagree_up_vs_inv = np.zeros((len(subjects),1)) #upright vs inverted

#Circles
circle_count = np.zeros((len(subjects),1)) #array of total circle correctly detected 
circle_total = np.zeros((len(subjects),1)) #array of total circles presented
false_alarm_subject = np.zeros((len(subjects),2,2,2,2)) #array of subjects circle false alarm detection with the count in appropriate condition


dprime_subjects = np.zeros((len(subjects),5))
dprime_diss = np.zeros((len(subjects),4))
pc_disagree_conditions = np.zeros((3, len(subjects))) #used for T-test at end of code


for j in range(len(subjects)):
    fname = path + subjects[j] + '_flip_detect.tab'
    data = read_tab(fname, group_start='trial_id', group_end='trial_ok', return_params=False)
    
    # t_mod: 0: not mod, 1: mod
    # m_mod 0: not mod, 1:mod
    # angle 0: normal, 1:flipped
    # match: 0: masker, 1: target
    correct_tracker = np.zeros((2,2,2,2))
    repmod_tracker = np.zeros((2,2,2,2))
    count_tracker = np.zeros((2,2,2,2))
    false_alarm_circle = np.zeros((2,2,2,2))
    
    trials_info = []
    trials_repmod = []
    trials_correct = []

    for i in range(len(data)):
        if data[i]['TRIAL PARAMETERS']:
            a = literal_eval(data[i]['TRIAL PARAMETERS'][0][0])['T_MOD']
            b = literal_eval(data[i]['TRIAL PARAMETERS'][0][0])['M_MOD']
            c = literal_eval(data[i]['TRIAL PARAMETERS'][0][0])['ANGLE']
            d = literal_eval(data[i]['TRIAL PARAMETERS'][0][0])['MATCH']
        #exclude circle trials & circle false alarms from analysis
            if literal_eval(data[i]['TRIAL PARAMETERS'][0][0])['CIRCLE_P']==1:
                circle_total[j][0] +=1 #total circles shown
                if literal_eval(data[i]['TRIAL PARAMETERS'][0][0])['RESPONSE']==3:
                    circle_count[j][0]+=1 #detected circle
            elif literal_eval(data[i]['TRIAL PARAMETERS'][0][0])['RESPONSE']==5:
                false_alarm_circle[a, b, c, d]+=1 #falsely detected circle
        #include trials without circle    
            else:   
                if literal_eval(data[i]['TRIAL PARAMETERS'][0][0])['SCORE']==1:
                    correct_tracker[a, b, c, d]+=1
                if literal_eval(data[i]['TRIAL PARAMETERS'][0][0])['RESPONSE']==1:
                    repmod_tracker[a, b, c, d]+=1
                count_tracker[a, b, c, d]+=1
                trials_info += [[a, b, c, d]]
                trials_repmod += [literal_eval(data[i]['TRIAL PARAMETERS'][0][0])['RESPONSE']]
                trials_correct += [literal_eval(data[i]['TRIAL PARAMETERS'][0][0])['SCORE']]
    
    false_alarm_subject[j]=false_alarm_circle #false alarms in order of subjects, number of false alarms for each condition

#part of your code
    n_trials = int(count_tracker.sum())
    trials_info = np.array(trials_info)
    trials_info = np.concatenate((np.ones((n_trials, 1)), trials_info), 1)
    trials_repmod = np.array(trials_repmod)
    trials_correct = np.array(trials_correct)
    
    pc = correct_tracker / count_tracker * 100
    pm = repmod_tracker / count_tracker * 100
    
    # %% Consider only when targ_mod and mask_mod disagree
    pm_disagree = pm[[0, 1], [1, 0]]
    pc_disagree = (100 - pm_disagree[0] + pm_disagree[1]) / 2
    
    hr = pm_disagree[1]
    mr = 100 - pm_disagree[1]
    fa = pm_disagree[0]
    cr = 100 - pm_disagree[0]
    
#d-prime arrays
    dp_disagree = dprime(np.transpose([hr, mr, fa, cr], [1, 2, 0]))

    dprime_diss[j][0] = dp_disagree[0, 1] #upright target
    dprime_diss[j][1] = dp_disagree[0, 0] #upright masker
    dprime_diss[j][2] = dp_disagree[1, 1] #flipped target
    dprime_diss[j][3] = dp_disagree[1, 0] #flipped masker

    # dprime difference array for subjects
    dprime_subjects[j][0] = (dp_disagree[0, 1] - dp_disagree[0, 0]) #upright T - upright M
    dprime_subjects[j][1]= (dp_disagree[0, 1] - dp_disagree[1, 1]) #upright T - flipped T
    dprime_subjects[j][2]=(dp_disagree[0, 0] - dp_disagree[1, 0]) #upright M - flipped M
    dprime_subjects[j][3]=((dp_disagree[0, 1] - dp_disagree[0, 0]) - (dp_disagree[1, 1] - dp_disagree[1, 0])) #upright vs inverted
    dprime_subjects[j][4]= (dp_disagree[1, 1] - dp_disagree[1, 0]) #inverted T - inverted M
    
    
#percent correct arrays   
    disagree_upright[j][0] = pc_disagree[0, 1] #upright target
    disagree_upright[j][1] = pc_disagree[0, 0] #upright masker   
    
    disagree_inverted[j][0] = pc_disagree[1, 1] # flipped target
    disagree_inverted[j][1] = pc_disagree[1, 0] #flipped masker
    
    disagree_tvm[j][0] = (pc_disagree[0, 1] - pc_disagree[0, 0]) #upright target - upright masker
    disagree_tvm[j][1] = (pc_disagree[1, 1] - pc_disagree[1, 0]) #flipped target - flipped masker
    
    disagree_up_vs_inv[j][0] = (pc_disagree[0, 1] - pc_disagree[0, 0])-(pc_disagree[1, 1] - pc_disagree[1, 0]) #upright - inverted
    

    pc_disagree_conditions[0][j]= (pc_disagree[0, 1] - pc_disagree[0, 0]) #upright target - upright masker
    pc_disagree_conditions[1][j]=(pc_disagree[1, 1] - pc_disagree[1, 0]) #inverted target - inverted masker
    pc_disagree_conditions[2][j]=(pc_disagree[0, 1] - pc_disagree[0, 0]) - (pc_disagree[1, 1] - pc_disagree[1, 0]) #upright - inverted

circle_pc = circle_count/circle_total #circle detection PC per subject
color_up = u'#999933'
color_down = u'#882255'
marker_up = '^-'
marker_down = 'v-'
ms=6

# Figure 1: PC Target vs Masker, Upright 
width = 5
height = 4
fname = 'Percent Correct of Upright Face Matching Target or Masker'
plt.figure(figsize=(width, height))
plt.plot(disagree_upright.T, 'o-', ms=8, alpha=0.5, zorder=-50)
performance = [disagree_upright.mean(0)[0]-50,disagree_upright.mean(0)[1]-50]
performanceSE = [disagree_upright.std(0)[0] / np.sqrt(len(subjects)),disagree_upright.std(0)[1] / np.sqrt(len(subjects))]
plt.bar((0,1), performance, width=0.4, bottom=50, align='center', alpha=1, edgecolor='black', facecolor = 'None', linewidth = 1.5, yerr=performanceSE, zorder=100) 
plt.xticks([0, 1], [0, 1])
plt.gca().set_xticklabels(['target', 'masker'])
plt.xlabel('Talker in Video')
plt.ylabel('Percent Correct')
plt.title('Percent Correct of Upright Face Matching Target or Masker')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(fname + '.png', dpi=300)

# Figure 2: PC Target vs Masker, Inverted 
width = 5
height = 4
fname = 'Percent Correct of Inverted Face Matching Target or Masker'
plt.figure(figsize=(width, height))
plt.plot(disagree_inverted.T, 'o-', ms=8, alpha=0.5, zorder=-50)
performance = [disagree_inverted.mean(0)[0]-50,disagree_inverted.mean(0)[1]-50]
performanceSE = [disagree_inverted.std(0)[0] / np.sqrt(len(subjects)),disagree_inverted.std(0)[1] / np.sqrt(len(subjects))]
plt.bar((0,1), performance, width=0.4, bottom=50, align='center', alpha=1, edgecolor='black', facecolor = 'None', linewidth = 1.5, yerr=performanceSE, zorder=100) 
plt.xticks([0, 1], [0, 1])
plt.gca().set_xticklabels(['target', 'masker'])
#plt.plot([0, 1], [0, 0], '--k', zorder=-100)
plt.xlabel('Talker in Video')
plt.ylabel('Percent Correct')
plt.title('Percent Correct of Inverted Face Matching Target or Masker')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(fname + '.png', dpi=300)
#plt.close('all')
# %%
width = 3.5
height = 3
fname = 'percor_target_masker_up_in'
barw=.2
plt.figure(figsize=(width, height))
#plt.plot((0-barw/2,1-barw/2), disagree_upright.T, marker_up, c=color_up, ms=ms-3, alpha=0.5, zorder=-50)
performance = [disagree_upright.mean(0)[0]-50,disagree_upright.mean(0)[1]-50]
performanceSE = [disagree_upright.std(0)[0] / np.sqrt(len(subjects)),disagree_upright.std(0)[1] / np.sqrt(len(subjects))]
plt.bar((0-barw/2,1-barw/2), performance, width=barw, bottom=50, align='center', alpha=1, edgecolor=color_up, facecolor = 'w')
plt.errorbar(x=(0-barw/2,1-barw/2), y=np.array(performance)+50, c=color_up, fmt=marker_up, ms=ms, mec=color_up, mfc='w', capsize=3, yerr=performanceSE) 
#plt.plot((0+barw/2,1+barw/2), disagree_inverted.T, marker_down, c=color_down, ms=ms-3, alpha=0.5, zorder=-50)
performance = [disagree_inverted.mean(0)[0]-50,disagree_inverted.mean(0)[1]-50]
performanceSE = [disagree_inverted.std(0)[0] / np.sqrt(len(subjects)),disagree_inverted.std(0)[1] / np.sqrt(len(subjects))]
plt.bar((0+barw/2,1+barw/2), performance, width=barw, bottom=50, align='center', alpha=1, edgecolor=color_down, facecolor = 'w')
plt.errorbar(x=(0+barw/2,1+barw/2), y=np.array(performance)+50, c=color_down, fmt=marker_down, ms=ms, mec=color_down, mfc='w', capsize=3, yerr=performanceSE)
plt.xticks([0, 1], [0, 1])
plt.gca().set_xticklabels(['Target', 'Masker'])
plt.xlabel(u'Talker in video')
plt.ylabel(u'Performance \n(% Correct)')
#plt.title('Audio-Visual Performance Benefit Across SNR')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(fname + '.pdf', dpi=600)
# %%
width = 3.5
height = 2
fname = 'percor_target_masker_up'
barw=.3
plt.figure(figsize=(width, height))
plt.subplot(121)
data_color='#DDDDDD'
color_up_light='#CCCC99'
color_down_light = '#C491AA'
[plt.plot(np.array([0,1]) + .1 * (np.random.rand() - 0.5), d, marker_up, c=color_up_light, ms=ms-5, lw=.5, zorder=-50) for d in  disagree_upright]
performance = [disagree_upright.mean(0)[0]-50,disagree_upright.mean(0)[1]-50]
performanceSE = [disagree_upright.std(0)[0] / np.sqrt(len(subjects)),disagree_upright.std(0)[1] / np.sqrt(len(subjects))]
plt.bar((0,1), performance, width=barw, bottom=50, align='center', alpha=1, edgecolor=color_up, facecolor = 'w', zorder=-100)
plt.errorbar(x=(0,1), y=np.array(performance)+50, c=color_up, fmt=marker_up, ms=ms, mec=color_up, mfc='w', capsize=3, yerr=performanceSE) 
plt.plot([-1, 2], [50, 50], 'k--', lw=.5, zorder=-200)
plt.xlim([-.25, 1.25])
#plt.plot((0+barw/2,1+barw/2), disagree_inverted.T, marker_down, c=color_down, ms=ms-3, alpha=0.5, zorder=-50)
#performance = [disagree_inverted.mean(0)[0]-50,disagree_inverted.mean(0)[1]-50]
#performanceSE = [disagree_inverted.std(0)[0] / np.sqrt(len(subjects)),disagree_inverted.std(0)[1] / np.sqrt(len(subjects))]
#plt.bar((0+barw/2,1+barw/2), performance, width=barw, bottom=50, align='center', alpha=1, edgecolor=color_down, facecolor = 'w')
#plt.errorbar(x=(0+barw/2,1+barw/2), y=np.array(performance)+50, c=color_down, fmt=marker_down, ms=ms, mec=color_down, mfc='w', capsize=3, yerr=performanceSE)
plt.xticks([0, 1], [0, 1])
plt.ylim([45, 100])
plt.yticks([50, 75, 100])
plt.gca().set_xticklabels(['Target', 'Masker'])
plt.xlabel(u'Talker in video')
plt.ylabel(u'Performance \n(% Correct)')
plt.title('Upright')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.subplot(122)
#plt.plot((0-barw/2,1-barw/2), disagree_upright.T, marker_up, c=color_up, ms=ms-3, alpha=0.5, zorder=-50)
#performance = [disagree_upright.mean(0)[0]-50,disagree_upright.mean(0)[1]-50]
#performanceSE = [disagree_upright.std(0)[0] / np.sqrt(len(subjects)),disagree_upright.std(0)[1] / np.sqrt(len(subjects))]
#plt.bar((0-barw/2,1-barw/2), performance, width=barw, bottom=50, align='center', alpha=1, edgecolor=color_up, facecolor = 'w')
#plt.errorbar(x=(0-barw/2,1-barw/2), y=np.array(performance)+50, c=color_up, fmt=marker_up, ms=ms, mec=color_up, mfc='w', capsize=3, yerr=performanceSE) 
[plt.plot(np.array([0,1]) + .1 * (np.random.rand() - 0.5), d, marker_down, c=color_down_light, ms=ms-5, lw=.5, zorder=-50) for d in disagree_inverted]
performance = [disagree_inverted.mean(0)[0]-50,disagree_inverted.mean(0)[1]-50]
performanceSE = [disagree_inverted.std(0)[0] / np.sqrt(len(subjects)),disagree_inverted.std(0)[1] / np.sqrt(len(subjects))]
plt.bar((0, 1), performance, width=barw, bottom=50, align='center', alpha=1, edgecolor=color_down, facecolor = 'w', zorder=-100)
plt.errorbar(x=(0,1), y=np.array(performance)+50, c=color_down, fmt=marker_down, ms=ms, mec=color_down, mfc='w', capsize=3, yerr=performanceSE)
plt.plot([-1, 2], [50, 50], 'k--', lw=.5, zorder=-200)
plt.xlim([-.25, 1.25])
plt.xticks([0, 1], [0, 1])
plt.ylim([45, 100])
plt.yticks([50, 75, 100])
plt.gca().set_xticklabels(['Target', 'Masker'])
plt.xlabel(u'Talker in video')
#plt.ylabel(u'Performance \n(% Correct)')
plt.title('Inverted')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(fname + '.pdf', dpi=600)
# %%

# Figure 3: Target vs Masker, Upright & Inverted
width = 3.5
height = 2
fname = 'Benefit of Target Face vs Masker Face'
plt.figure(figsize=(width, height))
plt.plot(disagree_tvm.T, '-', ms=ms-5, lw=.5, c=data_color, zorder=-600)
[plt.plot(0, d, marker_up, ms=ms-5, c=color_up_light, zorder=-50) for d in disagree_tvm.T[0]]
performance = [disagree_tvm.mean(0)[0],disagree_tvm.mean(0)[1]]
performanceSE = [disagree_tvm.std(0)[0] / np.sqrt(len(subjects)),disagree_tvm.std(0)[1] / np.sqrt(len(subjects))]
plt.bar((0), performance[0], width=0.4, bottom = 0, align='center', alpha=1, edgecolor=color_up, facecolor = 'None', zorder=-100) 
plt.errorbar(x=(0), y=performance[0], c=color_up, fmt=marker_up, ms=ms, mec=color_up, mfc='w', capsize=3, yerr=performanceSE[0])
[plt.plot(1, d, marker_down, ms=ms-5, c=color_down_light, zorder=-50) for d in disagree_tvm.T[1]]
plt.bar((1), performance[1], width=0.4, bottom = 0, align='center', alpha=1, edgecolor=color_down, facecolor = 'None', zorder=-100) 
plt.errorbar(x=(1), y=performance[1], c=color_down, fmt=marker_down, ms=ms, mec=color_down, mfc='w', capsize=3, yerr=performanceSE[1])
plt.xticks([0,1], [0,1])
plt.plot([-1, 2], [0, 0], 'k--', lw=.5, zorder=-200)
plt.xlim([-.25, 1.25])
plt.gca().set_xticklabels(['Upright', 'Inverted'])
plt.xlabel('Talker in Video')
plt.ylabel(u'Improvement \n(Δ% Correct)')
#plt.title('Benefit of Target Face')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(fname + '.pdf', dpi=600)
# %%

# Figure 4: Upright vs Inverted
width = 3.5
height = 2
fname = 'Benefit of Upright Face vs Inverted Face'
plt.figure(figsize=(width, height))
plt.subplot(122)
plt.plot(disagree_up_vs_inv.T, 'sk', ms=ms-5, zorder=-50)
performance = [disagree_up_vs_inv.mean(0)]
performanceSE = [disagree_up_vs_inv.std(0) / np.sqrt(len(subjects))]
#plt.bar((0), performance, width=0.4, bottom=0, align='center', edgecolor='k', lw=1, facecolor ='w', zorder=-100) 
plt.errorbar(x=(-.1), y=performance, c='k', fmt='s', ms=ms, mec='k', mfc='w', capsize=3, yerr=performanceSE)
plt.plot([-1, 1], [0, 0], 'k--', lw=.5, zorder=-200)
plt.xticks([])
plt.xlim([-.25, .25])
plt.ylim([-17, 25])
plt.ylabel(u'Upright Face Improvement \n(Δ% Correct)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.subplot(121)
performance = [disagree_tvm.mean(0).mean(0)]
performanceSE = [disagree_tvm.mean(1).std(0) / np.sqrt(len(subjects))]
[plt.plot(0, d, 'sk', ms=ms-5, zorder=-50) for d in disagree_tvm.mean(1)]
#plt.bar((0), performance, width=0.4, bottom=0, align='center', edgecolor='k', lw=1, facecolor ='w', zorder=-100) 
plt.errorbar(x=(-.1), y=performance, c='k', fmt='s', ms=ms, mec='k', mfc='w', capsize=3, yerr=performanceSE)
plt.plot([-1, 1], [0, 0], 'k--', lw=.5, zorder=-200)
plt.xticks([])
plt.xlim([-.25, .25])
plt.ylim([-17, 25])
#plt.gca().set_xticklabels(['Upright Benefit', 'Target Benefit'])
#plt.xlabel('')
plt.ylabel(u'Target Face Improvement \n(Δ% Correct)')
#plt.title('Benefit of Upright Face vs Inverted Face')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(fname + '.pdf', dpi=600)

#t-tests and shapiro test

for i in range(3):
    a=pc_disagree_conditions[i]
    t = scipy.stats.ttest_1samp(a, 0, axis=0)
    s = scipy.stats.shapiro(a)
    print(t)
    print(s)

    

"""# %% do the linear model -- this stuff should prop
trials_int = np.array([trials_info[:, 1] & trials_info[:, 2],
                       trials_info[:, 1] & trials_info[:, 3],
                       trials_info[:, 1] & trials_info[:, 4],
                       trials_info[:, 2] & trials_info[:, 3],
                       trials_info[:, 2] & trials_info[:, 4],
                       trials_info[:, 3] & trials_info[:, 4]]).T
# trials_int = np.array([trials_info[:, 3] & trials_info[:, 4]]).T
trials_info_int = np.concatenate((trials_info, trials_int), axis=1)

w_repmod, res, rank, s = np.linalg.lstsq(trials_info, trials_repmod)
w_repmod_int, res, rank, s = np.linalg.lstsq(trials_info_int, trials_repmod)
w_correct, res, rank, s = np.linalg.lstsq(trials_info[:, [0, 3, 4]], trials_correct)


plt.subplot(131)
plt.plot(trials_info.dot(w_repmod), trials_repmod + np.random.rand(n_trials) * 0.1, '.')

plt.subplot(132)
plt.plot(trials_info_int.dot(w_repmod_int), trials_repmod + np.random.rand(n_trials) * 0.1, '.')

plt.subplot(133)
plt.plot(trials_info[:, [0, 3, 4]].dot(w_correct), trials_repmod + np.random.rand(n_trials) * 0.1, '.')
"""


