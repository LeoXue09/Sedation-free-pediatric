import os
import numpy as np
import dicom2nifti
import shutil
import h5py
import tables
import nibabel as nib
from sklearn.model_selection import KFold
import logging
import time
import pydicom as dicom
import pandas as pd
from datetime import datetime
import skimage.transform as skitran

def main():
	data_dir = '/media/data/uni/ultra_fast/data/AC_300_15_35_nii'
	inp_acq = '15s'
	tar_acq = '300s'
	SUV = True
	if SUV:
		save_path = '/media/data/uni/ultra_fast/data/h5_file/AC_{}_{}_SUV'.format(inp_acq,tar_acq)
	else:
		save_path = '/media/data/uni/ultra_fast/data/h5_file/AC_{}_{}'.format(inp_acq,tar_acq)
	source_dir = '/media/data/uni/ultra_fast/data/AC_300_15_35_dicom'
	downsample_order = 4
	scale = 2
	width = 112
	# find_max(data_dir, acq_sec, downsample_order=downsample_order, scale=scale, width=width, SUV=SUV)
	# raise ValueError
	cross_validation = False
	### 300s AC_PET--max: 7176554.0	NAC_PET--max: 3805270.75	ratio--max: 314.1737976074219	Z--max: 439
	### 300s SUV --- AC_PET--max: 255.35353088378906	NAC_PET--max: 135.74156188964844	ratio--max: 37.16859436035156	Z--max: 439
	### 15s SUV --- AC_PET--max: 204.31632833804022	NAC_PET--max: 109.27755737304688	ratio--max: 37.08840735554823	Z--max: 439
	### 15s --- AC_PET--max: 2459127.290661554	NAC_PET--max: 1315251.8695728711	ratio--max: 807.8818048744818	Z--max: 439

	# center_max = [7176554, 3805270*0.8, 314*0.9]
	center_max = [2459127, 110, 38]
	generate_h5_3d(data_dir, inp_acq, tar_acq, save_path=save_path, center_max=center_max, cross_validation=cross_validation, fold=None, downsample_order=downsample_order,scale=scale,width=width,SUV=SUV)
	# shuffle_h5_3d(save_path, cross_validation=cross_validation,width=width)

def get_SUV_ratio(pid):
	df = pd.read_csv('/media/data/uni/ultra_fast/data/Dicom_header/AC_300_15_35_dicom.csv')
	df['PID'] = df['PID'].str.replace(' ','_')
	SUV_ratio = df[df['PID']==pid]['weight'].values * 1000 / df[df['PID']==pid]['Dose'].values
	# print(pid, SUV_ratio)
	return SUV_ratio.astype(float)

def find_max(data_dir, acq_sec, downsample_order, scale, width, SUV):
	z_max,ac_max, nac_max,ratio_max = 0,0,0,0
	count = 1 
	for pid in [x for x in os.listdir(data_dir) if not x.startswith('.')]:
		print('{}/{}--{}--start'.format(count, len(os.listdir(data_dir)), pid))
		count += 1
		nac_temp, ac_temp, z_temp = 0,0,0
		nac, ac = None, None
		nac = np.array(nib.load(os.path.join(data_dir,pid,'NASC_{}.nii.gz'.format(acq_sec))).dataobj).astype(float)
		if SUV:
			SUV_ratio = get_SUV_ratio(pid)
			nac *= SUV_ratio
		nac_post = cut_pad_downsample(nac, downsample_order, scale, width)
		nac_temp = np.max(nac_post)
		if nac_temp > nac_max:
			nac_max = nac_temp
		z_temp = nac.shape[2]
		if z_temp > z_max:
			z_max = z_temp
		ac = np.array(nib.load(os.path.join(data_dir,pid,'ASC_{}.nii.gz'.format(acq_sec))).dataobj).astype(float)
		if SUV:
			ac *= SUV_ratio
		ac_post = cut_pad_downsample(ac, downsample_order, scale, width)
		ac_temp = np.max(ac_post)
		if ac_temp > ac_max:
			ac_max = ac_temp
		ratio = compute_ratio(nac,ac,SUV)
		ratio_post = cut_pad_downsample(ratio, downsample_order, scale, width)
		ratio_temp = np.max(ratio_post)
		orr = np.count_nonzero(ratio_post>100)
		# print('ratio: ', ratio_post.shape, ratio_temp, orr)
		if ratio_temp > ratio_max:
			ratio_max = ratio_temp
		print('AC_PET--max: {}\tNAC_PET--max: {}\tratio--max: {}\tZ--max: {}'.format(ac_max, nac_max, ratio_max, z_max))
	# print('AC_PET--max: {}\tNAC_PET--max: {}\tratio--max: {}\tZ--max: {}'.format(ac_max, nac_max, ratio_max, z_max))

def compute_ratio(NAC, AC, SUV):
	input_x = NAC.copy()
	if not SUV:
		min_cutoff = 1
		
	else:
		min_cutoff = 1e-3
	input_x[input_x < min_cutoff] = 1
	out = AC/input_x
	out = np.clip(out, 1e-3, np.max(out))
	return out

def cut_pad_downsample(image, downsample_order, scale, width):
	img = image.copy()
	slice_num = int(img.shape[2])
	img = skitran.resize(img, (img.shape[0] // scale, img.shape[1] // scale, img.shape[2] // (scale*2)), order=downsample_order)

	pre_pad = int((width-img.shape[2])/2)
	out = np.pad(img, ((0,0),(0,0),(pre_pad,width-img.shape[2]-pre_pad)),'constant', constant_values=(0, 0))
	return out

def minmax_normalization_center(img, center_max):
	center_min = 0.0
	return (img - center_min) / (center_max - center_min)

def generate_h5_3d(data_dir, inp_acq, tar_acq, save_path, center_max, cross_validation, fold, downsample_order,scale,width,SUV):
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	def write_h5(target_data_dir,dataset,target_file,pid_idx):
		# AC_PET_pool = target_file.create_earray(target_file.root,dataset+'_AC_PET',tables.Float32Atom(),
  #                                     			shape=(0,width,width,width),expectedrows=1000000)
		NAC_PET_pool = target_file.create_earray(target_file.root,dataset+'_NAC_PET',tables.Float32Atom(),
                                      			shape=(0,96,96,width),expectedrows=1000000)
		ratio_pool = target_file.create_earray(target_file.root,dataset+'_ratio',tables.Float32Atom(),
                                      			shape=(0,96,96,width),expectedrows=1000000)
		filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
											shape=(0,1),expectedrows=1000000)

		count = 0
		for pid in pid_idx:
			print('{}--start'.format(pid))
			tar_nac = np.array(nib.load(os.path.join(data_dir,pid,'NASC_{}.nii.gz'.format(tar_acq))).dataobj).astype(float)
			tar_ac = np.array(nib.load(os.path.join(data_dir,pid,'ASC_{}.nii.gz'.format(tar_acq))).dataobj).astype(float)
			inp_nac = np.array(nib.load(os.path.join(data_dir,pid,'NASC_{}.nii.gz'.format(inp_acq))).dataobj).astype(float)
			if SUV:
				SUV_ratio = get_SUV_ratio(pid)
				tar_ac *= SUV_ratio
				tar_nac *= SUV_ratio
				inp_nac *= SUV_ratio
			ratio = compute_ratio(tar_nac,tar_ac,SUV)
			# ac = cut_pad_downsample(ac, downsample_order,scale,width)
			nac = cut_pad_downsample(inp_nac, downsample_order,scale,width)
			ratio = cut_pad_downsample(ratio, downsample_order,scale,width)

			nac = minmax_normalization_center(nac, center_max[1])
			# ac = minmax_normalization_center(ac, center_max[0])
			ratio = minmax_normalization_center(ratio, center_max[2])

			# AC_PET_pool.append(np.expand_dims(ac, axis=0))
			print(np.max(nac), np.min(nac), np.max(ratio), np.min(ratio))
			NAC_PET_pool.append(np.expand_dims(nac, axis=0))
			ratio_pool.append(np.expand_dims(ratio, axis=0))
			filenames_pool.append(np.expand_dims([pid], axis=0))
			count += 1
			print(os.path.basename(target_data_dir)+'__'+dataset+': {}/{}'.format(count, len(pid_idx)))

	patient_list = [x for x in os.listdir(data_dir) if not x.startswith('.')]

	if cross_validation == True:
		pass

	elif cross_validation == False:
		target_file = tables.open_file(os.path.join(save_path, 'data.h5'), mode='w')
		train_pid = patient_list[:30]
		valid_pid = list(set(patient_list)-set(train_pid))
		write_h5(data_dir, 'train', target_file, train_pid)
		write_h5(data_dir, 'valid', target_file, valid_pid)
		target_file.close()

	elif cross_validation == None:
		target_file = tables.open_file(os.path.join(save_path, 'data.h5'), mode='w')
		test_pid = patient_list
		write_h5(data_dir, 'test', target_file, test_pid)
		target_file.close()

	else:
		pass

def shuffle_h5_3d(h5_dir, cross_validation, width):
	def shuffle(dataset, data, target_file):
		shape = data.get(dataset + '_NAC_PET').shape
		# AC_PET_pool = target_file.create_earray(target_file.root,dataset+'_AC_PET',tables.Float32Atom(),
  #                                     			shape=(0,width,width,width),expectedrows=1000000)
		NAC_PET_pool = target_file.create_earray(target_file.root,dataset+'_NAC_PET',tables.Float32Atom(),
                                      			shape=(0,width,width,width),expectedrows=1000000)
		ratio_pool = target_file.create_earray(target_file.root,dataset+'_ratio',tables.Float32Atom(),
                                      			shape=(0,width,width,width),expectedrows=1000000)
		filenames_pool = target_file.create_earray(target_file.root,dataset+'_filenames',tables.StringAtom(itemsize = 100),
											shape=(0,1),expectedrows=1000000)

		index = np.arange(0, shape[0])
		if dataset == 'train':
			np.random.seed(1314)
			np.random.shuffle(index)

		count = 0
		for idx in index:
			count += 1
			print('\t'+dataset+': {}/{}'.format(count, len(index)))
			# AC_PET_pool.append(np.expand_dims(data.get(dataset+'_AC_PET')[idx,:,:,:], axis=0))
			NAC_PET_pool.append(np.expand_dims(data.get(dataset+'_NAC_PET')[idx,:,:,:], axis=0))
			ratio_pool.append(np.expand_dims(data.get(dataset+'_ratio')[idx,:,:,:], axis=0))
			filenames_pool.append(np.expand_dims(data.get(dataset+'_filenames')[idx], axis=0))

	for (dirpath, dirnames, filenames) in os.walk(h5_dir):
		for name in filenames:
			if name.endswith('.h5'):
				h5_path = os.path.join(dirpath, name)
				data = h5py.File(h5_path, mode='r')
				save_path = os.path.join(dirpath, os.path.splitext(name)[0]+'_shuffled.h5')
				target_file = tables.open_file(save_path,mode='w')
				if cross_validation == True:
					shuffle('train',data,target_file)
					shuffle('valid',data,target_file)
				else:
					shuffle('train',data,target_file)
				data.close()
				target_file.close()


if __name__ == '__main__':
	main()