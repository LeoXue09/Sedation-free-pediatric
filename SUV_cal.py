# -*- coding: utf-8 -*-
import h5py
import nibabel as nib
import os
import sys
import numpy as np
import pandas as pd
import time
import logging
import shutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pydicom as dicom
from scipy import spatial
from matplotlib import cm
from scipy.interpolate import interpn
from sklearn.linear_model import SGDRegressor
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
import skimage.transform as skitran

class Evaluate():
	def __init__(self):
		self.source_nii_dir = '/media/data/uni/ultra_fast/data/AC_300_15_35_nii'
		self.seg_dir = '/media/data/uni/ultra_fast/data/lesion_seg'
		self.result_dir = '/media/data/uni/ultra_fast/result'
		self.evaluate_dir_names = ['AC_15s','AC_15s_300s_SUV','AC_300s_SUV']
		self.evaluate_dir_list = [os.path.join(self.result_dir,x) for x in self.evaluate_dir_names]
		self.dicom_header = '/media/data/uni/ultra_fast/data/Dicom_header/AC_300_15_35_dicom.csv'
		self.save_dir = os.path.join(self.result_dir,'SUV_result')

	def calculate_SUV_features(self, PET, SUV, seg, pid):
		PET *= SUV
		pid_df = pd.DataFrame()
		for label in range(1, np.max(seg)+1):
			PET_voxels = []
			idx = np.argwhere(seg == label)
			for i in idx:
				x,y,z = i
				PET_voxels.append(PET[x,y,z])
			temp_df = pd.DataFrame({'PID':[pid], 'Label':[label], 'SUV Mean': [np.mean(PET_voxels)], 'SUV Max': [np.max(PET_voxels)]})
			temp_df = temp_df[['PID', 'Label', 'SUV Mean', 'SUV Max']]
			pid_df = pd.concat([pid_df, temp_df])
		return pid_df

	def calculate_single(self):
		PET_path = '/Users/songxue/Desktop/ultra_fast_result/non_sedate/ori/ZHAO_YI_JIN_PET104362_111958/609_n-15s-no_correction.nii.gz'
		seg_path = '/Users/songxue/Desktop/ultra_fast/Data/non_sedate_lesion/ZHAO_YI_JIN_300sCTC_sedation_seg.nii.gz'
		PET = np.array(nib.load(PET_path).dataobj)
		seg = np.array(nib.load(seg_path).dataobj)
		SUV = 8.0 * 1000 / 83250000.0
		PET *= SUV
		label_dic = {'liver':1, 'lesion':2}
		df = pd.DataFrame()
		for label in range(1, np.max(seg)+1):
			PET_voxels = []
			idx = np.argwhere(seg == label)
			for i in idx:
				x,y,z = i
				PET_voxels.append(PET[x,y,z])
			temp_df = pd.DataFrame({'Label':[label], 'SUV Mean': [np.mean(PET_voxels)], 'SUV Max': [np.max(PET_voxels)]})
			temp_df = temp_df[['Label', 'SUV Mean', 'SUV Max']]
			df = pd.concat([df, temp_df])
		save_path = os.path.join('/Users/songxue/Desktop/ultra_fast_result/non_sedate/SUV_result', os.path.basename(seg_path).split('.nii.gz')[0])
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		df.to_csv(os.path.join(save_path, os.path.basename(PET_path).split('.nii.gz')[0]+'.csv'),index=False)

	def for_ori(self):
		save_path = os.path.join(self.save_dir,'ori')
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		df = pd.read_csv(self.dicom_header)
		df['PID'] = df['PID'].str.replace(' ','_')
		test_pid_list = [x for x in os.listdir(self.evaluate_dir_list[0]) if not x.startswith('.')]
		for file in ['ASC_15s.nii.gz', 'ASC_300s.nii.gz', 'NASC_15s.nii.gz', 'NASC_300s.nii.gz']:
			SUV_df = pd.DataFrame()
			for count, pid in enumerate([x.split('.nii.gz')[0] for x in os.listdir(self.seg_dir) if not x.startswith('.')]):
				print('{}/{}--{}--start'.format(count+1,26,pid))
				SUV = df[df['PID']==pid]['weight'].values * 1000 / df[df['PID']==pid]['Dose'].values
				PET = np.array(nib.load(os.path.join(self.source_nii_dir,pid,file)).dataobj)
				seg = np.array(nib.load(os.path.join(self.seg_dir,'{}.nii.gz'.format(pid))).dataobj)
				temp_df = self.calculate_SUV_features(PET,SUV,seg,pid)
				SUV_df = pd.concat([SUV_df, temp_df])
				SUV_df.to_csv(os.path.join(save_path, '{}.csv'.format(file.split('.nii.gz')[0])),index=False)

	def for_gen(self):
		save_path = os.path.join(self.save_dir,'gen')
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		df = pd.read_csv(self.dicom_header)
		df['PID'] = df['PID'].str.replace(' ','_')
		test_pid_list = [x for x in os.listdir(self.evaluate_dir_list[0]) if not x.startswith('.')]
		for i, evaluate_dir in enumerate([x for x in self.evaluate_dir_list]):
			SUV_df = pd.DataFrame()
			for pid in [x for x in os.listdir(os.path.join(evaluate_dir,'individual')) if not x.startswith('.')]:
				print('{}--start'.format(pid))
				SUV = df[df['PID']==pid]['weight'].values * 1000 / df[df['PID']==pid]['Dose'].values
				for file in os.listdir(os.path.join(evaluate_dir,'individual',pid)):
					if 'AC_gen' in file:
						PET = np.array(nib.load(os.path.join(evaluate_dir,'individual',pid,file)).dataobj)
				seg = np.array(nib.load(os.path.join(self.seg_dir,'{}.nii.gz'.format(pid))).dataobj)
				pid_df = self.calculate_SUV_features(PET,SUV,seg,pid)
				SUV_df = pd.concat([SUV_df, pid_df])
				SUV_df.to_csv(os.path.join(save_path, '{}.csv'.format(self.evaluate_dir_names[i])),index=False)



if __name__ == '__main__':
	eva = Evaluate()
	# eva.for_ori()
	# eva.for_gen()
	eva.calculate_single()