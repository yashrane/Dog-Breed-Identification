import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import pickle
from skimage import io, transform
from PIL import Image
from DogsDataset import DogsDataset
import csv



# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
	'train': transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
}

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))



use_gpu = torch.cuda.is_available()


def train():
	datasets = {x: DogsDataset('labels/' + x + '.csv', 'images/', data_transforms[x]) for x in ['train', 'val']}
	data_loaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=4,shuffle=True, num_workers=4) 
		for x in ['train', 'val']}
	
	dataset_sizes ={x: len(datasets[x]) for x in ['train', 'val']}

	model_ft = models.resnet18(pretrained=True)
	num_ftrs = model_ft.fc.in_features

	model_ft.fc = nn.Linear(num_ftrs, 120)#probably want to replace this with softmax or something later


	if use_gpu:
		model_ft = model_ft.cuda()
	criterion = nn.CrossEntropyLoss()

	# Observe that all parameters are being optimized
	optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

	# Decay LR by a factor of 0.1 every 7 epochs
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

	model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,	dataset_sizes,data_loaders,
						   num_epochs=25)


	save_checkpoint({
			'state_dict': model_ft.state_dict(),
		})	
					   
	output = open('data.pkl', 'wb')

	# Pickle dictionary using protocol 0.
	pickle.dump(model_ft, output)



	




def train_model(model, criterion, optimizer, scheduler,dataset_sizes,data_loaders, num_epochs=25, resume=''):

	resume_epoch=0
	if resume:
		if os.path.isfile(resume):
			print("=> loading checkpoint '{}'".format(resume))
			checkpoint = torch.load(resume)
			resume_epoch = checkpoint['epoch']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
				.format(resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(resume))



	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs-resume_epoch):
		print('Epoch {}/{}'.format(epoch+resume_epoch, num_epochs - 1))
		print('-' * 10)

		for phase in ['train', 'val']:
			if phase == 'train':
				scheduler.step()
				model.train(True)  # Set model to training mode
			else:
				model.train(False)  # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0
	
			
			# Iterate over data.
			batch_num = 1
			for data in data_loaders[phase]:
				# get the inputs
				inputs = data[0]
				labels = data[1]

				if(batch_num%100 == 0):
					print("Batch#" + str(batch_num))
				batch_num+=1;
				
				# wrap them in Variable
				if use_gpu:
					inputs = Variable(inputs.cuda())
					labels = Variable(labels.cuda())
				else:
					inputs, labels = Variable(inputs), Variable(labels)

				# zero the parameter gradients
				optimizer.zero_grad()

				
				# forward
				outputs = model(inputs)
				_, preds = torch.max(outputs.data, 1)
				
				
				loss = criterion(outputs, labels)

				# backward + optimize only if in training phase
				if phase == 'train':
					loss.backward()
					optimizer.step()

				# statistics
				running_loss += loss.data[0] * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects / dataset_sizes[phase]

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))

			# deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

			save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'optimizer' : optimizer.state_dict(),
			})	
			
		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model
	
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

		
		
		
def test(resume_file):
	model = models.resnet18(pretrained=True)
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, 120)#probably want to replace this with softmax or something later
	
	
	best_wts = torch.load(resume_file)
	model.load_state_dict(best_wts['state_dict'])
	
	dataset = DogsDataset(csv_file = 'test/test_ids.csv', root_dir = 'test/',
							transform = data_transforms['val'], mode='test')
	data_loader = torch.utils.data.DataLoader(dataset, batch_size=4,shuffle=True, num_workers=4)
	
	test_model(model, data_loader, len(dataset))
	
	
		
def test_model(model, data_loader, data_size):

	model.train(False)  # Set model to evaluate mode
	with open('predictions.csv', 'w') as prediction_file:
		csvwriter = csv.writer(prediction_file)
		num_preds = 0
		for data in data_loader:
			if num_preds % 100 == 0:
				print('Predictions: {}/{}'.format(num_preds, len(data_loader)-1))
				print('-' * 10)
				print('')
			
			inputs = data[0]
			ids = data[1]
			
			# wrap them in Variable
			if use_gpu:
				inputs = Variable(inputs.cuda())
			else:
				inputs= Variable(inputs)
			
			outputs = model(inputs)
#			preds = F.softmax(outputs, dim=0).data
			_, preds = torch.max(outputs.data, 1)
			
			
			
			
			for i in range(len(ids)):
				id = ids[i]
			#	pred = preds[i].numpy()
				pred = np.zeros(120)
				pred[preds[i]] = 1
				
				row = [id]+pred.tolist()
				csvwriter.writerow(row)
			
			num_preds+=1


def visualize_model(model,dataloader,class_names, num_images=6):
	'''Function to visualize the output of a given model
	Args:
		model: The model to be visualized
		dataloader: the data to feed into the model
		num_images (optional): the number of images to visualize. Defaults to 6.
	'''

	images_so_far = 0
	#fig = plt.figure()

	for data in dataloader:
		inputs, ids = data
		if use_gpu:
			inputs = Variable(inputs.cuda())
		else:
			inputs= Variable(inputs)

		outputs = model(inputs)
		_, preds = torch.max(outputs.data, 1)

		for j in range(inputs.size()[0]):
			images_so_far += 1
			#ax = plt.subplot(num_images//2, 2, images_so_far)
			#ax.axis('off')
#			ax.set_title('predicted: {}'.format(class_names[preds[j]]))
#			img = unorm(inputs.data[j]).view(224,224,3)
			img_name = os.path.join('test_images/',
				ids[j]+'.jpg')
			img = Image.open(img_name)
			plt.imshow(img),plt.title('predicted: {}'.format(class_names[preds[j]]))
#			plt.imshow(img)
			print('wrote prediction#'+ str(images_so_far) )
			plt.savefig('predictions/prediction#' + str(images_so_far) + '.jpg')
			if images_so_far == num_images:
				#plt.savefig('predictions/predictions.jpg')
				return

def visualize(resume_file):
	model = models.resnet18(pretrained=True)
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, 120)#probably want to replace this with softmax or something later
	
	
	best_wts = torch.load(resume_file)
	model.load_state_dict(best_wts['state_dict'])
	
	dataset = DogsDataset('test_images/test_ids.csv', 'test_images/', data_transforms['val'], mode='test') 
	class_names = pd.read_csv('class_names.csv')['id']
	
	
	data_loader = torch.utils.data.DataLoader(dataset, batch_size=4,shuffle=True, num_workers=4)
	
	
	visualize_model(model, data_loader, class_names, num_images=10)

#train()			
#test('checkpoint.pth.tar')
visualize('checkpoint.pth.tar')