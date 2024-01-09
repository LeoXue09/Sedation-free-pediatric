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
	# organize_dir = '/media/data/uni/ultra_fast/data/AC_300_15_35_dicom'
	organize_dir = '/media/data/uni/ultra_fast/data/Child_new_0831'
	# read_dicom_header(organize_dir)
	convert_dcm2nii(organize_dir)
	# converted_dir = '/media/data/uni/ultra_fast/data/AC_300_15_35_nii'
	# rename_converted_nii(converted_dir)
	
def read_dicom_header(organize_dir):
	print('Reading Dicom header')
	FMT = '%H%M%S'
	save_dir = os.path.join(os.path.dirname(organize_dir), 'Dicom_header')
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	df = pd.DataFrame()
	for pid in [x for x in os.listdir(organize_dir) if not x.startswith('.')]:
		print('\t{}'.format(pid))
		for sub in [x for x in os.listdir(os.path.join(organize_dir,pid)) if not x.startswith('.')]:
			if '15S-Normal' in sub:
				tar_dir = os.path.join(organize_dir,pid,sub)
				if int(sub.split('_')[-1])>500:
					state = 'anesthesia'
				else:
					state = 'non_anesthesia'
				try:
					ds = dicom.read_file(os.path.join(tar_dir,os.listdir(tar_dir)[10]))
					ManufacturerModelName = ds[0x0008, 0x1090].value
					dose = int(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
					inj_start_time = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
					inj_start_time = inj_start_time.split('.')[0]
					acquisition_time = ds[0x0008, 0x0032].value.split('.')[0]
					post_inj_time = datetime.strptime(acquisition_time, FMT) - datetime.strptime(inj_start_time, FMT)
					days, seconds = post_inj_time.days, post_inj_time.seconds
					post_inj_minute = seconds // 60
					radionuclide_name = ds[0x0054, 0x0016][0][0x0018, 0x0031].value
					gender = ds[0x0010, 0x0040].value
					try:
						age = int(ds[0x0010, 0x1010].value[1:-1])
					except:
						age = None
					weight = float(ds[0x0010, 0x1030].value)
					height = float(ds[0x0010, 0x1020].value)
					BMI = weight / (height**2)
					df = df.append({'PID': pid, 'State': state, 'ManufacturerModelName': ManufacturerModelName, 'Dose': dose,
									'post_inj_time': post_inj_minute, 'radionuclide_name': radionuclide_name,
									'gender': gender, 'age': age, 'weight': weight, 'height': height, 'BMI': BMI}, ignore_index=True)
				except (AttributeError, KeyError):
					print('{}---error'.format(pid))
					continue
						
				df = df[['PID', 'State', 'ManufacturerModelName', 'Dose', 'post_inj_time', 'radionuclide_name',
						'gender', 'age', 'weight', 'height','BMI']]
				df.to_csv(os.path.join(save_dir,'{}.csv'.format(os.path.basename(organize_dir))), index=False)

def convert_dcm2nii(organize_dir):
	print('converting')
	for pid in [x for x in os.listdir(organize_dir) if not x.startswith('.')]:
		save_pid = pid.replace(' ', '_')
		save_dir = os.path.join(organize_dir.split('_dicom')[0]+'_nii',save_pid)
		if not os.path.exists(save_dir):
			print(pid)
			os.makedirs(save_dir)			
			for sub in [x for x in os.listdir(os.path.join(organize_dir,pid)) if not x.startswith('.')]:
				# if 'NC' in sub:
				print('\t{}'.format(sub))
				dicom2nifti.convert_directory(os.path.join(organize_dir,pid,sub), save_dir, compression=True, reorient=False)

def rename_converted_nii(converted_dir):
	print('Renaming')
	for pid in [x for x in os.listdir(converted_dir) if not x.startswith('.')]:
		print('\t{}'.format(pid))
		save_dir = os.path.join(os.path.dirname(converted_dir),'test',pid)
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		for file in [x for x in os.listdir(os.path.join(converted_dir,pid)) if not x.startswith('.')]:
			filename = file.split('.nii.gz')[0]
			ser = filename.split('_')[0]
			sec = filename.split('_')[1]
			mod = filename.split('_')[2]
			if mod == 'normal':
				outname = '{}_{}.nii.gz'.format('ASC',sec)
			elif mod == 'no':
				mod_ex = filename.split('_')[3]
				if mod_ex == 'scatter':
					outname = '{}_{}.nii.gz'.format('NSC',sec)
				elif mod_ex == 'correction':
					outname = '{}_{}.nii.gz'.format('NASC',sec)
			else:
				outname = '{}.nii.gz'.format('CT')

			shutil.copyfile(os.path.join(converted_dir,pid,file), os.path.join(save_dir,outname))
			# shutil.move(os.path.join(converted_dir,pid,file), os.path.join(converted_dir,pid,outname))


if __name__ == '__main__':
	organize_data()