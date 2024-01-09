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

class Resemble():
	def __init__(self):
		self.save_dir = '/media/data/uni/ultra_fast/data/test'
		self.data_path = '/media/data/uni/ultra_fast/data/h5_file/AC_300s/data.h5'
		self.domain = 'train'

	def resemble_from_h5(self):
		
		data = h5py.File(self.data_path, 'r')
		filenames = np.array(data[self.domain+'_filenames']).flatten()
		filenames = [x.decode('utf-8') for x in filenames]

		h5_nac = np.array(data.get(self.domain+'_NAC_PET')[0])
		h5_ratio = np.array(data.get(self.domain+'_ratio')[0])
		pid = filenames[0]
		save_path = os.path.join(self.save_dir,pid)
		if not os.path.exists(save_path):
			os.makedirs(save_path)

		nib.save(nib.Nifti1Image(h5_nac, affine=np.eye(4)),
				 os.path.join(save_path, 'h5_nac.nii.gz'))

		nib.save(nib.Nifti1Image(h5_ratio, affine=np.eye(4)),
				 os.path.join(save_path, 'h5_ratio.nii.gz'))
		data.close()

if __name__ == '__main__':
	eva = Resemble()
	eva.resemble_from_h5()
