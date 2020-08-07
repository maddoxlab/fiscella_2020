#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:18:43 2018

@author: sarafiscella
"""

import sys #fixed error of expyfun module
import numpy as np
import matplotlib.pyplot as plt #contains imread
from expyfun import (ExperimentController, fetch_data_file, analyze as ea,
                     building_doc)
import random
from expyfun.visual import (Text, RawImage)
from expyfun.stimuli import (play_sound, rms, window_edges)
import cv2
from expyfun.io import (read_wav, read_tab)
from cv2 import VideoCapture, imread
import pyglet
from pyglet.image import AbstractImage
import pandas as pd
from pandas import DataFrame
from ast import literal_eval
import csv 
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from expyfun.io import (write_hdf5, read_hdf5)

plt.rc('font', size=18)
plt.rc('axes', titlesize=24)

subjects = ['003', '004', '005', '006', '007', '008', '009', '010', '011', '012','013', '014', 
            '015', '016', '017']
angles = np.array([0, 90, 180])
snrs = np.array([0, -3, -6])
num_correct = np.zeros((len(subjects), 4, len(snrs), 192/12))
num_presented = np.zeros((len(subjects), 4, len(snrs), 192/12))

percent_correct = np.zeros((len(subjects), 4, len(snrs), 192/12))
counts = np.zeros((len(subjects), 4, len(snrs)))

for i,sub in enumerate(subjects):
    
    fname = '/Users/sarafiscella/Documents/Python/Flippedfaceexperiment/Flipface_data/' + sub + '.csv'
    with open(fname, 'rU') as csvfile:
       reader = csv.reader(csvfile, delimiter=',')
       data = [r for r in reader][1:]
    
    for d in data:
        if d[6]== 'TRUE':
            angle_index = 3
        else:
            angle_index = np.where([abs(int(d[1]))==a for a in angles])[0][0]
        snr_index = np.where([(int(d[4]))==s for s in snrs])[0][0]
        num_correct[i, angle_index, snr_index, int(counts[i, angle_index, snr_index])] = d[-3]
        num_presented[i, angle_index, snr_index, int(counts[i, angle_index, snr_index])] = d[-2]
        #percent_correct[i, angle_index, snr_index, int(counts[i, angle_index, snr_index])] = d[-1]
        counts[i, angle_index, snr_index] += 1
    #percent_correct_condition = num_correct.sum(-1) / num_presented.sum(-1).astype(float)
    percent_correct = num_correct.sum(-1, keepdims=True) / num_presented.sum(-1, keepdims=True)
    relative_p_correct = np.zeros((15,3,3,1))
    relative_p_correct[:,0,:]=percent_correct[:,0,:]-percent_correct[:,-1,:]  
    relative_p_correct[:,1,:]=percent_correct[:,1,:]-percent_correct[:,-1,:]  
    relative_p_correct[:,2,:]=percent_correct[:,2,:]-percent_correct[:,-1,:]  



    plt.figure()
    for p in percent_correct[i]:
        per_mean = p.mean(-1)
        plt.plot(snrs, per_mean*100)
    plt.xlabel('SNR (db)')
    plt.ylabel('Percent Correct')
    plt.title(sub)
    plt.legend(('0', '90', '180', 'Static'))

    
percent_condition_subject = percent_correct.mean(-1)
mean_across_subjects = percent_condition_subject.mean(0)
sem = percent_condition_subject.std(axis=0)/np.sqrt(len(subjects))

write_hdf5('matrix1.hdf5', dict(data=percent_condition_subject), overwrite=True)

percent_across_snrs = percent_condition_subject.mean(2)
mean_across_subjects2 = percent_across_snrs.mean(0) 
sem2 = percent_across_snrs.std(axis=0)/np.sqrt(len(subjects))
    #plt.plot(snrs, mean_sub*100, 'o')

width = 8
height = 5
fname = 'Average Percent Correct Across SNR'

plt.figure(figsize=(width, height))
plt.errorbar(x=[0, 1, 2, 3], y=mean_across_subjects2*100, yerr=sem2*100, fmt='o')
plt.xticks([0, 1, 2, 3], [0, 1, 2, 3])
plt.gca().set_xticklabels(['0', '90', '180', 'Static'])
plt.xlabel('Angle (deg)')
plt.ylabel('Percent Correct')
plt.title('Average Percent Correct Across SNR')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(fname + '.png', dpi=300)

width = 8
height = 5
fname = 'Average Percent Correct'
plt.figure(figsize=(width, height))
for mean_sub, sem_std, xshift in zip(mean_across_subjects, sem, np.linspace(.1, -.1, 4)):
    plt.errorbar(x=snrs+xshift, y=mean_sub*100, yerr=sem_std*100, fmt='o-')
plt.xlabel('SNR (dB)')
plt.ylabel('Percent Correct')
plt.title('Average Percent Correct')
plt.legend((u'0°', u'90°', u'180°', 'Static'))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(fname + '.png', dpi=300)


plt.figure()
for mean_sub, sem_std in zip(mean_across_subjects, sem):
    plt.plot(snrs, mean_sub*100)
plt.xlabel('SNR (dB)')
plt.ylabel('Percent Correct')
plt.title('Average')
plt.legend(('0', '90', '180', 'Static'))

#plt.figure()
ang1=percent_across_snrs[:,0]
ang2=percent_across_snrs[:,1]
ang3=percent_across_snrs[:,2]
ang4=percent_across_snrs[:,3]
data = np.rec.array([('0', ang1[0]),
                     ('0', ang1[1]),
                     ('0', ang1[2]),
                     ('0', ang1[3]),
                     ('0', ang1[4]),
                     ('0', ang1[5]),
                     ('90', ang2[0]),
                     ('90', ang2[1]),
                     ('90', ang2[2]),
                     ('90', ang2[3]),
                     ('90', ang2[4]),
                     ('90', ang2[5]),
                     ('180', ang3[0]),
                     ('180', ang3[1]),
                     ('180', ang3[2]),
                     ('180', ang3[3]),
                     ('180', ang3[4]),
                     ('180', ang3[5]),
                     ('Static', ang4[0]),
                     ('Static', ang4[1]),
                     ('Static', ang4[2]),
                     ('Static', ang4[3]),
                     ('Static', ang4[4]),
                     ('Static', ang4[5])], dtype=[('Archer','|U5'),('Score', float)])
                     
                     
                     
f_oneway(ang1, ang2, ang3, ang4)
mc = MultiComparison(data['Score'], data['Archer'])
result = mc.tukeyhsd()


 
print(result)
print(mc.groupsunique)


rel_percent_condition_subject = relative_p_correct.mean(-1)
rel_mean_across_subjects = rel_percent_condition_subject.mean(0)
rel_sem = rel_percent_condition_subject.std(axis=0)/np.sqrt(len(subjects))

#write_hdf5('matrix1.hdf5', dict(data=percent_condition_subject), overwrite=True)

rel_percent_across_snrs = rel_percent_condition_subject.mean(2)
rel_mean_across_subjects2 = rel_percent_across_snrs.mean(0) 
rel_sem2 = rel_percent_across_snrs.std(axis=0)/np.sqrt(len(subjects))
    #plt.plot(snrs, mean_sub*100, 'o')

width = 8
height = 5
fname = 'Audio-Visual Performance Benefit Across SNR'

plt.figure(figsize=(width, height))
plt.errorbar(x=[0, 1, 2], y=rel_mean_across_subjects2*100, yerr=rel_sem2*100, fmt='o')
plt.xticks([0, 1, 2], [0, 1, 2])
plt.gca().set_xticklabels(['0', '90', '180'])
plt.xlabel('Angle (deg)')
plt.ylabel(u'Δ Percent Correct')
plt.title('Audio-Visual Performance Benefit Across SNR')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(fname + '.png', dpi=300)

width = 8
height = 5
fname = 'Audio-Visual Performance Benefit'
plt.figure(figsize=(width, height))
for mean_sub, sem_std, xshift in zip(rel_mean_across_subjects, rel_sem, np.linspace(.1, -.1, 4)):
    plt.errorbar(x=snrs+xshift, y=mean_sub*100, yerr=sem_std*100, fmt='o-')
plt.xlabel('SNR (dB)')
plt.ylabel(u'Δ Percent Correct')
plt.title('Audio-Visual Performance Benefit')
plt.legend((u'0°', u'90°', u'180°'))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(fname + '.png', dpi=300)



# Change the width and height to be what you want in inches and add a name
# make sure all figures are smaller than 8.5 x 11"


# Here is the code that does the plotting and labels axes goes #




    

