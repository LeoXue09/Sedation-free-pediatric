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
from skimage.measure import compare_ssim
import skimage.transform as skitran
import keras.backend as K
K.set_image_data_format("channels_first")
import tensorflow as tf

from L1_stride_3D import SR_UnetGAN
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

class Resemble():
	def __init__(self):
		self.loca = 'ultra_fast'
		self.acq_sec = '15s'
		self.SUV = True
		self.nii_data_path = '/media/data/uni/ultra_fast/data/AC_300_15_35_nii'
		self.source_dir = self.nii_data_path

		if self.SUV:
			self.data_path = '/media/data/uni/ultra_fast/data/h5_file/AC_{}_SUV/data.h5'.format(self.acq_sec)
			self.save_dir = '/media/data/uni/ultra_fast/result/AC_{}_SUV'.format(self.acq_sec)
		else:
			self.data_path = '/media/data/uni/ultra_fast/data/h5_file/AC_{}/data.h5'.format(self.acq_sec)
			self.save_dir = '/media/data/uni/ultra_fast/result/AC_{}'.format(self.acq_sec)
		# self.model_path = os.path.join(self.save_dir, 'model/generator_epoch_400.hdf5')

		self.data_path = '/media/data/uni/ultra_fast/data/Child_new_0831_nii'
		self.save_dir = '/media/data/uni/ultra_fast/result/Child_new_SUV'
		self.model_path = '/media/data/uni/ultra_fast/result/AC_15s_300s_SUV/model/generator_epoch_400.hdf5'

		self.generator = SR_UnetGAN().build_generator()
		self.generator.load_weights(self.model_path)

		self.upsample_order = 5
		self.scale = 2

		### 300s AC_PET--max: 7176554.0	NAC_PET--max: 3805270.75	ratio--max: 314.1737976074219	Z--max: 439
		### 300s SUV --- AC_PET--max: 255.35353088378906	NAC_PET--max: 135.74156188964844	ratio--max: 37.16859436035156	Z--max: 439
		### 15s SUV --- AC_PET--max: 204.31632833804022	NAC_PET--max: 109.27755737304688	ratio--max: 37.08840735554823	Z--max: 439
		### 15s --- AC_PET--max: 2459127.290661554	NAC_PET--max: 1315251.8695728711	ratio--max: 807.8818048744818	Z--max: 439
		self.center_max = [38, 110]
		# self.center_max = [807*0.9, 1315251*0.8]

	def de_normalization(self,volume, center_max):
		volume = np.array(volume)
		volume = np.clip(volume,0, np.max(volume))
		volume *= center_max
		return volume

	def get_SUV_ratio(self, pid):
		# df = pd.read_csv('/media/data/uni/ultra_fast/data/Dicom_header/AC_300_15_35_dicom.csv')
		df = pd.read_csv('/media/data/uni/ultra_fast/data/Dicom_header/Child_new_0831.csv')
		df['PID'] = df['PID'].str.replace(' ','_')
		SUV_ratio = df[(df['PID']==pid) & (df['State']=='non_anesthesia')]['weight'].values * 1000 / df[(df['PID']==pid) & (df['State']=='non_anesthesia')]['Dose'].values
		return SUV_ratio

	def minmax_normalization_center(self, img, center_max):
		center_min = 0.0
		out = (img - center_min) / (center_max - center_min)
		return np.clip(out, 0, 1)

	def cut_pad_downsample(self, image, downsample_order, width):
		print('\tDownsampling')
		img = image.copy()
		slice_num = int(img.shape[2])
		img = skitran.resize(img, (img.shape[0] // self.scale, img.shape[1] // self.scale, img.shape[2] // (self.scale*2)), order=downsample_order)
		pre_pad = int((width-img.shape[2])/2)
		lat_pad = width-img.shape[2]-pre_pad
		out = np.pad(img, ((0,0),(0,0),(pre_pad,lat_pad)),'constant', constant_values=(0, 0))
		return out, slice_num, pre_pad, lat_pad

	def upsample_apply_ratio_map(self, network_out, slice_num, NAC, pre_pad, lat_pad):
		print('\tUpsampling')
		low_res = network_out.copy()
		low_res = low_res[:,:,pre_pad:-lat_pad]
		out_ratio = skitran.resize(low_res, (low_res.shape[0] * self.scale, low_res.shape[1] * self.scale, low_res.shape[2] * (self.scale*2)), order=self.upsample_order)
		if out_ratio.shape[2] > NAC.shape[2]:
			out_ratio = out_ratio[:,:,-NAC.shape:]
		else:
			out_ratio = np.pad(out_ratio, ((0,0),(0,0),(NAC.shape[2]-out_ratio.shape[2],0)),'constant', constant_values=(0, 0))

		# 	min_cutoff = 1
		# else:
		# 	min_cutoff = 1e-3
		# input_x[input_x < min_cutoff] = 1

		# out_ratio[NAC < 1] = 1
		return NAC*out_ratio, out_ratio

	def resemble_from_nii(self):
		data = h5py.File(self.data_path, mode='r')
		valid_filenames = np.array(data['valid_filenames']).flatten()
		valid_filenames = [x.decode('utf-8') for x in valid_filenames]
		data.close()

		count = 0
		for pid in valid_filenames:
			count += 1
			print('{}/{}--{}--start'.format(count,len(valid_filenames),pid))
			save_path = os.path.join(self.save_dir,'individual',pid)
			if not os.path.exists(save_path):
				os.makedirs(save_path)

			file_nac = nib.load(os.path.join(self.nii_data_path,pid,'NASC_{}.nii.gz'.format(self.acq_sec)))
			affine = file_nac.affine
			nac = np.array(file_nac.dataobj)
			if self.SUV:
				SUV_ratio = self.get_SUV_ratio(pid)
				nac *= SUV_ratio

			input_nac, slice_num, pre_pad, lat_pad = self.cut_pad_downsample(nac, downsample_order=4, width=112)
			input_nac = self.minmax_normalization_center(input_nac, self.center_max[1])
			x_slice = np.expand_dims(input_nac, axis=0)
			x_slice = np.expand_dims(x_slice, axis=0)
			out_volume = self.generator.predict(x_slice)[0,0,:,:,:]
			out_volume = self.de_normalization(out_volume, self.center_max[0])
			out_ac, out_ratio = self.upsample_apply_ratio_map(out_volume, slice_num, nac, pre_pad, lat_pad)
			if self.SUV:
				SUV_ratio = self.get_SUV_ratio(pid)
				out_ac /= SUV_ratio
			nib.save(nib.Nifti1Image(out_ac, affine=affine),
					 os.path.join(save_path, 'AC_gen_{}.nii.gz'.format(self.acq_sec)))
			nib.save(nib.Nifti1Image(out_ratio, affine=affine),
					 os.path.join(save_path, 'ratio_gen_{}.nii.gz'.format(self.acq_sec)))
			print('\t{}--done'.format(pid))

	def resemble_from_extra(self):
		for pid in [x for x in os.listdir(self.data_path) if not x.startswith('.')]:
			print('{}--start'.format(pid))
			if pid == 'JIANG_YI_ZE_PET103552_124145':
				save_path = os.path.join(self.save_dir,'individual',pid)
				if not os.path.exists(save_path):
					os.makedirs(save_path)

				for file in [x for x in os.listdir(os.path.join(self.data_path,pid)) if not x.startswith('.')]:
					if '15s-no_correction' in file:
						filepath = os.path.join(self.data_path,pid,file)
						if int(file.split('_')[0])>500:
							state = 'anesthesia'
						else:
							state = 'non_anesthesia'

						file_nac = nib.load(filepath)
						affine = file_nac.affine
						nac = np.array(file_nac.dataobj)
						if self.SUV:
							SUV_ratio = self.get_SUV_ratio(pid)
							nac *= SUV_ratio

						input_nac, slice_num, pre_pad, lat_pad = self.cut_pad_downsample(nac, downsample_order=4, width=112)
						input_nac = self.minmax_normalization_center(input_nac, self.center_max[1])
						x_slice = np.expand_dims(input_nac, axis=0)
						x_slice = np.expand_dims(x_slice, axis=0)
						out_volume = self.generator.predict(x_slice)[0,0,:,:,:]
						out_volume = self.de_normalization(out_volume, self.center_max[0])
						out_ac, out_ratio = self.upsample_apply_ratio_map(out_volume, slice_num, nac, pre_pad, lat_pad)
						if self.SUV:
							SUV_ratio = self.get_SUV_ratio(pid)
							out_ac /= SUV_ratio
						nib.save(nib.Nifti1Image(out_ac, affine=affine),
								 os.path.join(save_path, 'AC_gen_{}.nii.gz'.format(state)))
						nib.save(nib.Nifti1Image(out_ratio, affine=affine),
								 os.path.join(save_path, 'ratio_gen_{}.nii.gz'.format(state)))
						print('\t{}--done'.format(pid))

if __name__ == '__main__':
	eva = Resemble()
	# eva.resemble_from_nii()
	eva.resemble_from_extra()
