import os
import numpy as np
import h5py
import glob
import skimage.io as ski
import nibabel as nib
import SimpleITK as sitk
from nibabel import processing
import shutil
import pandas as pd
import dicom2nifti
import pydicom as dicom
from datetime import datetime
import skimage
import skimage.transform as skitran
import scipy.io
from zipfile import ZipFile

def organize_data():
	organize_dir = '/Volumes/INTENSO/recon'
	save_dir = '/Users/songxue/Desktop/ultra_fast/Data/Original'
	source_dir = '/Users/songxue/Desktop/ultra_fast/Data/No_correction_converted'
	convert_dcm2nii(organize_dir,source_dir,save_dir)
	# rename_converted_nii(organize_dir)

def load_dicom_series(data_dir):
	np_PET = []
	for s in [x for x in os.listdir(data_dir) if not x.startswith('.')]:
		img = dicom.read_file(os.path.join(data_dir,s), force=True).pixel_array.astype('float32')
		np_PET.append(img)
	return np.array(np_PET)

def find_affine(source_dir, pid, file_string):
	for file in [x for x in os.listdir(os.path.join(source_dir,pid)) if not x.startswith('.')]:
		for s in file_string:
			if s == '5s':
				if '5s' in file and '15s' not in file:
					tar_path = os.path.join(source_dir,pid,file)
					out_file = file
			else:
				if s in file:
					tar_path = os.path.join(source_dir,pid,file)
					out_file = file
		else:
			pass
	return nib.load(tar_path), out_file

def convert_dcm2nii(organize_dir, source_dir, input_save_dir=None):
	print('converting')
	pid_list = [x for x in os.listdir(source_dir) if not x.startswith('.')]
	for pid in [x for x in os.listdir(organize_dir) if not x.startswith('.')]:
		if pid in pid_list:
			print(pid)
			if not input_save_dir:
				save_dir = os.path.join(organize_dir+'_converted',pid)
			else:
				save_dir = os.path.join(input_save_dir+'_converted',pid)
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)			
			for ser in [x for x in os.listdir(os.path.join(organize_dir,pid)) if not x.startswith('.')]:
				if '5MIN_' in ser:
					affine, file = find_affine(source_dir, pid, ['5min','300s'])
				elif '15S_' in ser:
					affine, file = find_affine(source_dir, pid, ['15s'])
				elif '5S_' in ser and '15S' not in ser:
					affine, file = find_affine(source_dir, pid, ['5s'])
				elif '10S_' in ser:
					affine, file = find_affine(source_dir, pid, ['10s'])
				else:
					continue

				img = load_dicom_series(os.path.join(organize_dir,pid,ser))
				img = np.moveaxis(img, 0, -1)
				print(img.shape, affine.dataobj.shape, ser, file)
				nib.save(nib.Nifti1Image(img, affine.affine), os.path.join(save_dir,file))

def rename_converted_nii(organize_dir):
	print('Renaming')
	converted_dir = organize_dir+'_converted'
	for pid in [x for x in os.listdir(converted_dir) if not x.startswith('.')]:
		print('\t{}'.format(pid))
		for file in [x for x in os.listdir(os.path.join(converted_dir,pid)) if not x.startswith('.')]:
			if '_ct_' in file:
				shutil.move(os.path.join(converted_dir,pid,file), os.path.join(converted_dir,pid,'CT.nii.gz'))
			elif file.endswith('_nac.nii.gz'):
				shutil.move(os.path.join(converted_dir,pid,file), os.path.join(converted_dir,pid,'NAC.nii.gz'))
			elif 'pet_wb' in file:
				shutil.move(os.path.join(converted_dir,pid,file), os.path.join(converted_dir,pid,'AC.nii.gz'))
			else:
				print("Unrecognized file: {}".format(file))




if __name__ == '__main__':
	organize_data()