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
os.environ["CUDA_VISIBLE_DEVICES"]="2" 

class Resemble():
	def __init__(self):
		self.loca = 'ultra_fast'
		self.nii_data_path = '/media/data/uni/ultra_fast/data/AC_300_15_35_nii'
		self.center_max = [34, 367250]
		self.source_dir = self.nii_data_path
		self.data_path = '/media/data/uni/ultra_fast/data/h5_file/AC_300s/data.h5'

		self.save_dir = '/media/data/uni/ultra_fast/result/NC_model'
		self.model_path = '/media/uni/ST_2/Attenuation/result/SH_Vision_ratio_downorder_4_112_prepad_new_weight/model/generator_epoch_100.hdf5'

		self.generator = SR_UnetGAN().build_generator()
		self.generator.load_weights(self.model_path)

		self.upsample_order = 5

	def de_normalization(self,volume, center_max):
		volume = np.array(volume)
		volume = np.clip(volume,0, np.max(volume))
		volume *= center_max
		return volume

	def minmax_normalization_center(self, img, center_max):
		center_min = 0.0
		out = (img - center_min) / (center_max - center_min)
		return np.clip(out, 0, 1)

	def cut_pad_downsample(self, image, downsample_order, scale, width):
		print('\tDownsampling')
		img = image.copy()
		### Adjusting slice thickness to be 2, which is consistent to SH_Vision
		slice_num = int(img.shape[2] // (2/2.886))
		img = skitran.resize(img, (img.shape[0], img.shape[1], slice_num), order=1)
		if img.shape[2] < 448:
			img = np.pad(img, ((0,0),(0,0),(0,448-img.shape[2])),'constant', constant_values=(0, 0))
		else:
			img = img[:,:,-448:]
		img = skitran.resize(img, (width-2, width-2, img.shape[2] // scale), order=downsample_order)
		out = np.pad(img, ((1,1),(1,1),(0,0)),'constant', constant_values=(0, 0))
		return out, slice_num

	def upsample_apply_ratio_map(self, network_out, slice_num, NAC):
		print('\tUpsampling')
		scale = 4
		low_res = network_out.copy()
		low_res = low_res[1:-1,1:-1,:]
		out_dim = 192
		out_slice_ratio = (2.886/2)
		high_res = skitran.resize(low_res, (out_dim, out_dim, low_res.shape[2] * scale), order=self.upsample_order)
		if slice_num <= 448:
			out_ratio = high_res[:, :, :slice_num]
			out_slice = NAC.shape[2]
		else:
			out_ratio = high_res[:, :, :]
			out_slice = int(out_ratio.shape[2] // out_slice_ratio)
		out_ratio = skitran.resize(out_ratio, (out_ratio.shape[0], out_ratio.shape[1], out_slice), order=1)
		nac = NAC[:,:,-out_slice:]
		out_ratio[nac < 1] = 1
		return nac*out_ratio, out_ratio

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
			for file in [x for x in os.listdir(os.path.join(self.nii_data_path,pid)) if not x.startswith('.')]:
				file_nac = nib.load(os.path.join(self.nii_data_path,pid,file))
				affine = file_nac.affine
				nac = np.array(file_nac.dataobj)
				# slice_num = nac.shape[2]
				input_nac, slice_num = self.cut_pad_downsample(nac, downsample_order=4, scale=4, width=112)
				input_nac = self.minmax_normalization_center(input_nac, self.center_max[1])
				x_slice = np.expand_dims(input_nac, axis=0)
				x_slice = np.expand_dims(x_slice, axis=0)
				out_volume = self.generator.predict(x_slice)[0,0,:,:,:]
				out_volume = self.de_normalization(out_volume, self.center_max[0])
				out_ac, out_ratio = self.upsample_apply_ratio_map(out_volume, slice_num, nac)
				nib.save(nib.Nifti1Image(out_ac, affine=affine),
						 os.path.join(save_path, file))
				print('\t{}--{}--done'.format(pid,file))

if __name__ == '__main__':
	eva = Resemble()
	eva.resemble_from_nii()
