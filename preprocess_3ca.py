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
import skimage.transform as skitran
from skimage.metrics import structural_similarity as compare_ssim
from matplotlib import pylab as plt

def load_dicom_series(data_dir):
	if os.path.exists(os.path.join(data_dir,'ddd.nii.gz')):
		out = np.array(nib.load(os.path.join(data_dir,'ddd.nii.gz')).dataobj)
		return np.moveaxis(out, -1, 0)
	else:
		np_PET = []
		for s in [x for x in os.listdir(data_dir) if not x.startswith('.')]:
			img = dicom.read_file(os.path.join(data_dir,s), force=True).pixel_array.astype('float32')
			np_PET.append(img)
		return np.array(np_PET)

def main():
	data_dir = '/Volumes/INTENSO/three_cases'
	save_dir = '/Users/songxue/Desktop/ultra_fast/Result/three_cases'
	count = 0
	pid_list = []
	d2_rmse, d100_rmse, d5_rmse, d10_rmse, d20_rmse, d30_rmse = [],[],[],[],[],[]
	d2_ssim, d100_ssim, d5_ssim, d10_ssim, d20_ssim, d30_ssim = [],[],[],[],[],[]
	for pid in [x for x in os.listdir(data_dir) if not x.startswith('.')]:
		print(pid)
		ori_img, d_2s, d_100s, d_5s, d_10s, d_20s = None,None,None,None,None,None
		for ser in [x for x in os.listdir(os.path.join(data_dir,pid)) if not x.startswith('.')]:
			if '5MIN_' in ser:
				ori_img = load_dicom_series(os.path.join(data_dir,pid,ser))
			elif '1MIN_' in ser:
				d_100s = load_dicom_series(os.path.join(data_dir,pid,ser))
			elif '2S_' in ser:
				d_2s = load_dicom_series(os.path.join(data_dir,pid,ser))
			elif '5S_' in ser:
				d_5s = load_dicom_series(os.path.join(data_dir,pid,ser))
			elif '10S_' in ser:
				d_10s = load_dicom_series(os.path.join(data_dir,pid,ser))
			elif '20S_' in ser:
				d_20s = load_dicom_series(os.path.join(data_dir,pid,ser))
			elif '30S_' in ser:
				d_30s = load_dicom_series(os.path.join(data_dir,pid,ser))

		min_z = min(ori_img.shape[0], d_5s.shape[0], d_10s.shape[0], d_20s.shape[0], d_30s.shape[0], d_2s.shape[0], d_100s.shape[0])
		ori_img = ori_img[-min_z:,:,:]
		d_5s = d_5s[-min_z:,:,:]
		d_10s = d_10s[-min_z:,:,:]
		d_20s = d_20s[-min_z:,:,:]
		d_30s = d_30s[-min_z:,:,:]
		d_2s = d_2s[-min_z:,:,:]
		d_100s = d_100s[-min_z:,:,:]
		
		rmse, ssim = compute_metrics(ori_img, d_5s)
		d5_rmse.append(rmse)
		d5_ssim.append(ssim)

		rmse, ssim = compute_metrics(ori_img, d_10s)
		d10_rmse.append(rmse)
		d10_ssim.append(ssim)

		rmse, ssim = compute_metrics(ori_img, d_20s)
		d20_rmse.append(rmse)
		d20_ssim.append(ssim)

		rmse, ssim = compute_metrics(ori_img, d_30s)
		d30_rmse.append(rmse)
		d30_ssim.append(ssim)

		rmse, ssim = compute_metrics(ori_img, d_2s)
		d2_rmse.append(rmse)
		d2_ssim.append(ssim)

		rmse, ssim = compute_metrics(ori_img, d_100s)
		d100_rmse.append(rmse)
		d100_ssim.append(ssim)

		pid_list.append(pid)

		d5_df = pd.DataFrame({'PID': pid_list, 'RMSE': d5_rmse, 'SSIM': d5_ssim})
		d5_df = d5_df[['PID','RMSE','SSIM']]
		d5_df.to_csv(os.path.join(save_dir,'d5_metrics.csv'), index=False)

		d10_df = pd.DataFrame({'PID': pid_list, 'RMSE': d10_rmse, 'SSIM': d10_ssim})
		d10_df = d10_df[['PID','RMSE','SSIM']]
		d10_df.to_csv(os.path.join(save_dir,'d10_metrics.csv'), index=False)

		d20_df = pd.DataFrame({'PID': pid_list, 'RMSE': d20_rmse, 'SSIM': d20_ssim})
		d20_df = d20_df[['PID','RMSE','SSIM']]
		d20_df.to_csv(os.path.join(save_dir,'d20_metrics.csv'), index=False)

		d30_df = pd.DataFrame({'PID': pid_list, 'RMSE': d30_rmse, 'SSIM': d30_ssim})
		d30_df = d30_df[['PID','RMSE','SSIM']]
		d30_df.to_csv(os.path.join(save_dir,'d30_metrics.csv'), index=False)

		d2_df = pd.DataFrame({'PID': pid_list, 'RMSE': d2_rmse, 'SSIM': d2_ssim})
		d2_df = d2_df[['PID','RMSE','SSIM']]
		d2_df.to_csv(os.path.join(save_dir,'d2_metrics.csv'), index=False)

		d100_df = pd.DataFrame({'PID': pid_list, 'RMSE': d100_rmse, 'SSIM': d100_ssim})
		d100_df = d100_df[['PID','RMSE','SSIM']]
		d100_df.to_csv(os.path.join(save_dir,'d100_metrics.csv'), index=False)


def compute_metrics(real, pred):
	real[real<1] = 0
	pred[real<1] = 0
	mse = np.mean(np.square(real-pred))
	rmse = np.sqrt(mse)
	real_norm = real / float(np.max(real))
	pred_norm = pred / float(np.max(pred))
	ssim = compare_ssim(real_norm, pred_norm)
	return rmse, ssim


if __name__ == '__main__':
	main()