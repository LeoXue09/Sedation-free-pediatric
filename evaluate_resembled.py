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
		# self.acq_sec = 'non_anesthesia'
		self.acq_sec = 'anesthesia'
		# self.source_nii_dir = '/media/data/uni/ultra_fast/data/AC_300_15_35_nii'
		self.source_nii_dir = '/media/data/uni/ultra_fast/data/Child_new_0831_nii'
		# self.evaluate_dir = '/media/data/uni/ultra_fast/result/AC_{}'.format(self.acq_sec)
		self.evaluate_dir = '/media/data/uni/ultra_fast/result/Child_new_SUV'
		# self.dicom_header = '/media/data/uni/ultra_fast/data/Dicom_header/AC_300_15_35_dicom.csv'
		self.save_dir = self.evaluate_dir

	def compute_metrics(self, real_input, pred_input):
		try:
			real = real_input.copy()
			# real[real<self.threshold] = 0
			pred = pred_input.copy()
			# pred[real<self.threshold] = 0
			mse = np.mean(np.square(real-pred))
			# nrmse = np.sqrt(np.sum(np.square(real-pred))/np.sum(np.square(real)))
			nrmse = np.sqrt(mse) / (np.max(real)-np.min(real))
			ok_idx = np.where(real!=0)
			mape = np.mean(np.abs((real[ok_idx] - pred[ok_idx]) / real[ok_idx]))
			PIXEL_MAX = np.max(real)
			psnr = 20*np.log10(PIXEL_MAX / np.sqrt(np.mean(np.square(real-pred))))
			real_norm = real / float(np.max(real))
			pred_norm = pred / float(np.max(pred))
			ssim = compare_ssim(real_norm, pred_norm)
		except ValueError:
			mse, nrmse, mape, psnr, ssim = 0,0,0,0,0
		return mse, nrmse, mape, psnr, ssim

	def create_heatmap(self, ac_mip, nac_mip, gen_ac_mip, save_path):
		def scale(img, input_min, input_max):
			return (img-input_min) / ((input_max-input_min))
		center = 0
		nac = nac_mip - ac_mip
		gen = gen_ac_mip - ac_mip
		mmax = max(np.max(nac), np.max(gen))
		mmin = min(np.min(nac), np.min(gen))

		neg_nac = scale(nac * (nac<center), mmin, 0) - 1
		neg_nac *= (nac<center)
		neg_gen = scale(gen * (gen<center), mmin, 0) - 1
		neg_gen *= (gen<center)

		pos_nac = scale(nac * (nac>center), 0, mmax)
		pos_nac *= (nac>center)
		pos_gen = scale(gen * (gen>center), 0, mmax)
		pos_gen *= (gen>center)
		
		cen_nac = nac * (nac==center)
		cen_gen = gen * (gen==center)
		
		out_nac = neg_nac+pos_nac+cen_nac
		out_gen = neg_gen+pos_gen+cen_gen

		fig = plt.figure(figsize=(15, 15))
		fig.add_subplot(1,2,1)
		plt.title('NAC_Heatmap')
		plt.imshow(out_nac, cmap=plt.cm.RdGy, vmin=-1, vmax=1)
		plt.colorbar()
		plt.axis('off')
		fig.add_subplot(1,2,2)
		plt.title('Gen_Heatmap')
		plt.imshow(out_gen, cmap=plt.cm.RdGy, vmin=-1, vmax=1)
		plt.colorbar()
		plt.axis('off')
		plt.savefig(os.path.join(save_path, 'MIP_heatmap.png'), bbox_inches='tight')
		plt.close()

	def createMIP(self, img, modality, save_path):
		mip = np.amax(img[:,:,:], axis=1)
		rotate = np.rot90(mip)
		return rotate

	def evalaute(self):
		eva_path = os.path.join(self.evaluate_dir,'individual')
		general_save_path = os.path.join(self.save_dir,'general_result')
		if not os.path.exists(general_save_path):
			os.makedirs(general_save_path)
		gen_ac_list = [x for x in os.listdir(eva_path) if not x.startswith('.')]
		pids, mse_list, nrmse_list, mape_list, psnr_list, ssim_list, similarity_list = [],[],[],[],[],[],[]
		ori_mse_list, ori_nrmse_list, ori_mape_list, ori_psnr_list, ori_ssim_list, ori_similarity_list = [],[],[],[],[],[]
		for count, pid in enumerate(gen_ac_list):
			print('{}/{}--{}--start'.format(count+1,len(gen_ac_list),pid))
			individual_save_path = os.path.join(eva_path,pid)
			gen_ac = np.array(nib.load(os.path.join(eva_path,pid,'AC_gen_{}.nii.gz'.format(self.acq_sec))).dataobj)
			# ori_ac = np.array(nib.load(os.path.join(self.source_nii_dir,pid,'ASC_{}.nii.gz'.format(self.acq_sec))).dataobj)
			# ori_nac = np.array(nib.load(os.path.join(self.source_nii_dir,pid,'NASC_{}.nii.gz'.format(self.acq_sec))).dataobj)

			for file in [x for x in os.listdir(os.path.join(self.source_nii_dir,pid)) if not x.startswith('.')]:
				if self.acq_sec == 'non_anesthesia':
					if int(file.split('_')[0])<500:
						if 'n-15s-normal.nii.gz' in file:
							ori_ac = np.array(nib.load(os.path.join(self.source_nii_dir,pid,file)).dataobj)
						elif '15s-no_correction.nii.gz' in file:
							ori_nac = np.array(nib.load(os.path.join(self.source_nii_dir,pid,file)).dataobj)
				else:
					if int(file.split('_')[0])>500:
						if 'n-15s-normal.nii.gz' in file:
							ori_ac = np.array(nib.load(os.path.join(self.source_nii_dir,pid,file)).dataobj)
						elif '15s-no_correction.nii.gz' in file:
							ori_nac = np.array(nib.load(os.path.join(self.source_nii_dir,pid,file)).dataobj)

			# print('\tPlotting MIP & Heatmap')
			# ac_mip = self.createMIP(ori_ac, 'AC', individual_save_path)
			# nac_mip = self.createMIP(ori_nac, 'NAC', individual_save_path)
			# gen_ac_mip = self.createMIP(gen_ac, 'gen_AC', individual_save_path)
			# self.create_heatmap(ac_mip, nac_mip, gen_ac_mip, individual_save_path)
			print('\tCalculating metrics')
			mse, nrmse, mape, psnr, ssim = self.compute_metrics(ori_ac, gen_ac)
			# similarity = self.plot_dvh(ori_ac, gen_ac, ori_nac, pid, individual_save_path, modality='PRED')
			ori_mse, ori_nrmse, ori_mape, ori_psnr, ori_ssim = self.compute_metrics(ori_ac, ori_nac)
			# ori_similarity = self.plot_dvh(ori_ac, ori_nac, ori_nac, pid, individual_save_path, modality='NAC')
			print(ori_nrmse, nrmse)
			mse_list.append(mse)
			nrmse_list.append(nrmse* 1e2)
			psnr_list.append(psnr)
			ssim_list.append(ssim)
			mape_list.append(mape)
			ori_mse_list.append(ori_mse)
			ori_nrmse_list.append(ori_nrmse* 1e2)
			ori_psnr_list.append(ori_psnr)
			ori_ssim_list.append(ori_ssim)
			ori_mape_list.append(ori_mape)
			pids.append(pid)

			df = pd.DataFrame({'PID': pids, 'MSE': mse_list, 'ori_MSE': ori_mse_list, 'NRMSE %': nrmse_list, 
							   'ori_NRMSE %': ori_nrmse_list, 'MAPE': mape_list, 'ori_MAPE': ori_mape_list, 
							   'PSNR': psnr_list, 'ori_PSNR': ori_psnr_list, 'SSIM': ssim_list, 'ori_SSIM': ori_ssim_list})
			df = df.append({'PID': 'Mean Value', 'MSE': np.mean(mse_list),'ori_MSE': np.mean(ori_mse_list),
							'NRMSE %': np.mean(nrmse_list), 'ori_NRMSE %': np.mean(ori_nrmse_list),
							'MAPE': np.mean(mape_list),'ori_MAPE': np.mean(ori_mape_list),
							'PSNR': np.mean(psnr_list), 'ori_PSNR': np.mean(ori_psnr_list),
							'SSIM': np.mean(ssim_list), 'ori_SSIM': np.mean(ori_ssim_list)}, ignore_index=True)
			df = df[['PID', 'MSE','ori_MSE', 'MAPE','ori_MAPE','NRMSE %','ori_NRMSE %', 'PSNR','ori_PSNR', 'SSIM','ori_SSIM']]
			df.to_csv(os.path.join(general_save_path,'{}.csv'.format(self.acq_sec)), index=False)

	def get_SUV_ratio(self, pid):
		if 'Bern' in self.loca:
			dirs = os.path.join(self.source_dicom_dir,pid,'AC_PET')
		elif 'SH' in self.loca:
			dirs = os.path.join(self.source_dicom_dir,pid,'AC')
		for file in [x for x in os.listdir(dirs) if not x.startswith('.')]:
			try:
				ds = dicom.read_file(os.path.join(dirs,file))
			except:
				continue
			else:
				weight = float(ds[0x0010, 0x1030].value) * 1000
				dose = float(ds[0x0054, 0x0016][0][0x0018, 0x1074].value)
				break
		out_ratio = weight/dose
		return out_ratio

	def plot_histogram(self, mask, image, SUV_ratio, organ, modality, save_path):
		save_path = os.path.join(save_path, 'Organ_histogram')
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		img = image.copy()
		img *= SUV_ratio
		voxels = []
		idx = np.argwhere(mask != 0)
		for i in idx:
			x,y,z = i
			voxels.append(img[x,y,z])
		voxels = np.array(voxels)
		counts, bins = np.histogram(voxels.ravel(),bins=50)
		fig2, ax2 = plt.subplots()
		ax2.set_title('{} SUV Histogram'.format(organ))
		ax2.hist(bins[:-1], bins, weights=counts)
		fig2.savefig(os.path.join(save_path, '{}_{}_SUV_Histogram.png'.format(organ, modality)),bbox_inches='tight')

	def compute_DVH_points(self, mask, img, scaling=False, ori_max=None, stops=None):
		interval = 100
		voxels = []
		idx = np.argwhere(mask != 0)
		for i in idx:
			x,y,z = i
			voxels.append(img[x,y,z])
		voxels = np.array(voxels)
		if scaling == True:
			fac = ori_max / np.max(voxels)
			voxels = voxels * fac
		volumn = np.size(voxels)
		if stops is None:
			stops = np.arange(0,np.max(voxels)*(1+2/interval),np.max(voxels)/interval)
		x, y = [],[]
		for s in stops:
			cut = np.size(np.where(voxels >= s)[0])
			y.append(round(cut/volumn*100,1))
			x.append(round(s,1))
		return x,y, np.max(voxels), stops

	def plot_dvh(self, ori_dose, pre_dose, ori_nac, pid, save_path, cut_slice, modality):
		print('\tPlotting DVH---{}'.format(modality))
		save_path = os.path.join(self.save_dir,'individual_result', self.loca, pid, 'organ')
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		organ_list = ['liver', 'spleen', 'left_kidney', 'right_kidney']
		mask_label_list = [3, 13, 6, 7]
		color_list = ['red','green','blue','orange']
		fig1, ax1 = plt.subplots()
		ax1.set_title('Dose Volume Histogram')
		max_dic = {}
		if self.loca in ['SH_GE', 'SH_UI']:
			SUV_ratio = 1
		else:
			SUV_ratio = self.get_SUV_ratio(pid)

		try:
			all_mask = np.array(nib.load(os.path.join(self.source_nii_dir,'{}/seg.nii'.format(pid))).dataobj)
		except:
			print('{} without mask'.format(pid))
			return 1
		else:
			similarity = []
			for i, organ in enumerate(organ_list):
				if self.loca == 'Quadra':
					all_mask = all_mask[:,:,:cut_slice]
				elif 'Vision' in self.loca:
					all_mask = all_mask[:,:,-cut_slice:]
				mask = all_mask.copy()
				mask[np.where(mask>(mask_label_list[i]+0.001))] = 0
				mask[np.where(mask<(mask_label_list[i]-0.001))] = 0
				try:
					x1,y1,ori_max, stops = self.compute_DVH_points(mask, ori_dose, scaling=False)
				except:
					continue
				else:
					max_dic[organ] = ori_max
					ax1.plot(x1,y1,':',color=color_list[i])
					x2,y2,_,_ = self.compute_DVH_points(mask, pre_dose, scaling=False, ori_max=max_dic[organ], stops=stops)
					ax1.plot(x2,y2,'-',color=color_list[i])
					similarity.append(1 - spatial.distance.cosine(y1,y2))
					if modality == 'PRED':
						print('\t\tPlotting Histogram---{}'.format(organ))
						self.plot_histogram(mask, ori_dose, SUV_ratio, organ, 'AC', save_path)
						self.plot_histogram(mask, pre_dose, SUV_ratio, organ, 'AC_gen', save_path)
						self.plot_histogram(mask, ori_nac, SUV_ratio, organ, 'NAC', save_path)
			ax1.set_xlabel('Dose (Bq/mL)')
			ax1.set_ylabel('Volume percentage (%)')
			if modality == 'NAC':
				fig1.savefig(os.path.join(save_path, 'DVH_nac.png'),bbox_inches='tight')
			elif modality == 'PRED':
				fig1.savefig(os.path.join(save_path, 'DVH_pred.png'),bbox_inches='tight')
			plt.close()
			return np.mean(similarity)

	def normal_equation(self, x, y):
		print('\tNormal equation')
		# X = x.reshape(-1,1)
		x_bias = np.ones((len(x),1))
		x = np.reshape(x,(len(x),1))
		x = np.append(x_bias,x,axis=1)
		x_transpose = np.transpose(x)
		x_transpose_dot_x = x_transpose.dot(x)
		print('\t\tinversing')
		temp_1 = np.linalg.inv(x_transpose_dot_x)
		temp_2 = x_transpose.dot(y)
		theta = temp_1.dot(temp_2)
		theta_shaped = np.reshape(theta,(len(theta),1))
		y_hat = np.dot(x,theta_shaped)
		y_hat = y_hat.flatten()
		r2 = np.corrcoef(y_hat, y)[0, 1]**2
		return theta, r2

	def plot_scatterheat(self):
		eva_path = os.path.join(self.evaluate_dir,self.loca)
		gen_ac_list = [x for x in os.listdir(eva_path) if os.path.isfile(os.path.join(eva_path,x))]
		for count, file in enumerate(gen_ac_list):
			pid = file.split('_ac_gen')[0]
			print('{}/{}--{}--start'.format(count+1,len(gen_ac_list),pid))
			gen_ac = np.array(nib.load(os.path.join(eva_path,file)).dataobj)
			if self.loca in ['SH_GE', 'SH_UI']:
				gen_ac *= self.scanner_correction_ratio
			cut_slice = gen_ac.shape[2]
			ori_ac, ori_nac = self.find_ac_affine(pid, cut_slice)
			if 'Vision' in self.loca:
				downsample_scale = 4
			elif self.loca == 'SH_GE':
				downsample_scale = 2
			elif self.loca == 'SH_UI':
				downsample_scale = 1
			print('\tDownsampling')
			gen_ac = self.downsample(gen_ac, downsample_order=4, scale=downsample_scale)
			ori_ac = self.downsample(ori_ac, downsample_order=4, scale=downsample_scale)
			ori_nac = self.downsample(ori_nac, downsample_order=4, scale=downsample_scale)
			print('\tPlotting Scatterheat---PRED')
			save_path = os.path.join(self.save_dir,'individual_result', self.loca, pid)
			if not os.path.exists(save_path):
				os.makedirs(save_path)
			x = ori_ac.flatten()
			X = x.reshape(-1,1)
			y1 = gen_ac.flatten()
			y2 = ori_nac.flatten()
			print('\t\tHistogramming')
			data , x_e, y_e = np.histogram2d(x, y1, bins = 20, density = True)
			print('\t\tInterpolating')
			z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y1]).T , method = "splinef2d", bounds_error = False)
			#To be sure to plot all data
			z[np.where(np.isnan(z))] = 0.0
			# Sort the points by density, so that the densest points are plotted last
			idx = z.argsort()
			sort_x, sort_y1, z = x[idx], y1[idx], z[idx]
			print('\t\tPlotting scatter')
			fig, ax1 = plt.subplots(figsize=[10,10])
			divider = make_axes_locatable(ax1)
			cax = divider.append_axes('right', size='5%', pad=0.05)
			im = ax1.scatter(sort_x, sort_y1, c=z, s=10, cmap='rainbow', alpha=0.5)
			ax1.set_ylim(0, max(x))
			ax1.set_xlim(0, max(x))
			ax1.set_aspect('equal', adjustable='box')
			fig.colorbar(im, cax=cax, orientation='vertical')
			# plt.savefig(os.path.join(save_path, 'PRED_scatterheat_m: {},b: {}.png'.format(m,b)), bbox_inches='tight')
			fig.savefig(os.path.join(save_path, 'PRED_scatterheat.png'), bbox_inches='tight')
			plt.close()
			print('\tPlotting Scatterheat---NAC')
			print('\t\tHistogramming')
			data , x_e, y_e = np.histogram2d(x, y2, bins = 20, density = True)
			print('\t\tInterpolating')
			z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y2]).T , method = "splinef2d", bounds_error = False)
			z[np.where(np.isnan(z))] = 0.0
			idx = z.argsort()
			sort_x, sort_y2, z = x[idx], y2[idx], z[idx]
			print('\t\tPlotting scatter')
			fig, ax1 = plt.subplots(figsize=[10,10])
			divider = make_axes_locatable(ax1)
			cax = divider.append_axes('right', size='5%', pad=0.05)
			im = ax1.scatter(sort_x, sort_y2, c=z, s=10, cmap='rainbow', alpha=0.5)
			ax1.set_ylim(0, max(x))
			ax1.set_xlim(0, max(x))
			ax1.set_aspect('equal', adjustable='box')
			fig.colorbar(im, cax=cax, orientation='vertical')
			# plt.savefig(os.path.join(save_path, 'NAC_scatterheat_m: {},b: {}.png'.format(m,b)), bbox_inches='tight')
			fig.savefig(os.path.join(save_path, 'NAC_scatterheat.png'), bbox_inches='tight')
			plt.close()
			print('\tFitting lines---PRED')
			theta1, r2_1 = self.normal_equation(x, y1)
			b1,w1 = theta1
			theta2, r2_2 = self.normal_equation(x, y2)
			b2, w2 = theta2
			df = pd.DataFrame({'PID': [pid], 'Pred_w': [w1], 'Pred_b': [b1], 'Pred_r2': [r2_1], 'NAC_w': [w2], 'NAC_b': [b2], 'NAC_r2': [r2_2]})
			df = df[['PID', 'Pred_w', 'Pred_b', 'Pred_r2', 'NAC_w', 'NAC_b', 'NAC_r2']]
			df.to_csv(os.path.join(save_path,'fitted_line_parameter.csv'), index=False)
			print(df)

	


if __name__ == '__main__':
	eva = Evaluate()
	eva.evalaute()
	# eva.evaluate_organ()
	# eva.plot_scatterheat()