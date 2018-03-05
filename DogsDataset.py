import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from skimage import io, transform
from PIL import Image

class DogsDataset(Dataset):

	def __init__(self, csv_file='', root_dir="/", transform=None, mode='train'):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.NUM_CLASSES = 120
		self.mode = mode
		
		self.labels = self.parseData(csv_file)
		self.root_dir = root_dir
		self.transform = transform

		
		

	def parseData(self, file_path):
		data = pd.read_csv(file_path)
		#selected_breed_list = list(data.groupby('breed').count().sort_values(by='id', ascending=False).head(self.NUM_CLASSES).index)
		#data = data[data['breed'].isin(selected_breed_list)]
		data['target'] = 1
		#data['rank'] = data.groupby('breed').rank()['id']
		if self.mode is 'train':
			data_pivot = data.pivot('id', 'breed', 'target').reset_index().fillna(0)
		else:
			data_pivot = data.pivot('id','target').reset_index().fillna(0)
		#data_pivot = data.pivot('id', 'breed').reset_index().fillna(0)
		#print(data_pivot['breed'].unique())
		
		
		
		return data_pivot

	def __len__(self):
		return len(self.labels)
	
	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir,
								self.labels.iloc[idx, 0]+'.jpg')
		image = Image.open(img_name)
		
		if self.mode is 'train':
			labels = self.labels.iloc[idx, 1:].as_matrix().astype('float32')
			label = np.argmax(labels)
		else:
			label = self.labels.iloc[idx, 0]
			
		if self.transform:
			image = self.transform(image)
		sample = (image,label)
#		sample = (image, labels)

		return sample
		
		
	
		
