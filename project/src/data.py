# Run this code to unzip the Skin Diseases Data into a "dataset" folder
import zipfile as zf
files = zf.ZipFile("archive.zip", 'r')
files.extractall('dataset')
files.close()

# Useful Imports

# Torch Imports
import torch 
import torch.nn as nn
import torch.utils.data as data

# Torchvision Imports
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# Utility Imports
import matplotlib.pyplot as plt
import numpy as np

# Metric and Matrix Imports
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, accuracy_score

# Bayesian Optimization Imports
import GPyOpt

# Converts every image to 256x256 dimension, randomly change images for generalization,
# into a tensor, normalize pixel values
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Converts every image to 256x256 dimension and into a Tensor
val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Loads each subset directory from your "dataset" folder
train_dataset = datasets.ImageFolder(root='dataset/Train', transform=train_transform)
test_dataset = datasets.ImageFolder(root='dataset/Test', transform=val_test_transform)
val_dataset = datasets.ImageFolder(root='dataset/Val', transform=val_test_transform)

# Hyper Parameters for dataset organization and processing
batch_size = 25
num_workers = 3
# Dataloader for each subset which will batch, shuffle, and parallel load the data
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Total images in each subset
print('Train data: ', len(train_dataset))
print('Test data: ', len(test_dataset))
print('Valid data: ', len(val_dataset))

# distribution of classes across datasets:
# https://discuss.pytorch.org/t/how-to-see-distribution-of-labels-in-dataset/158379
from collections import Counter
train_labels = dict(Counter(train_dataset.targets))
val_labels = dict(Counter(val_dataset.targets))
test_labels = dict(Counter(test_dataset.targets))

print(train_labels.values())
print(val_labels.values())
print(test_labels.values())

diseases = ['Herpes', 'Melanoma', 'Monkeypox', 'Sarampion', 'Varicela']

fig_train, ax_train = plt.subplots()
p_train = ax_train.bar(diseases, train_labels.values())
ax_train.bar_label(p_train)
ax_train.set_ylabel('count')
ax_train.set_title('Train Data')
plt.show()

fig_val, ax_val = plt.subplots()
p_val = ax_val.bar(diseases, val_labels.values())
ax_val.bar_label(p_val)
ax_val.set_ylabel('count')
ax_val.set_title('Validation Data')
plt.show()

fig_test, ax_test = plt.subplots()
p_test = ax_test.bar(diseases, test_labels.values())
ax_test.bar_label(p_test)
ax_test.set_ylabel('count')
ax_test.set_title('Test Data')
plt.show()

# Check a batch of data
images, labels = next(iter(train_dataloader))

print(images[0].size()) # Number of Colors (RGB=3), Height=256, Width=256
print(labels[0]) # Herpes=0, Melanoma=1, Monkeypox=2, Sarampion=3, Varicela=4
print(images.shape)
print(labels.shape)

# Will show the image sampled
image = transforms.ToPILImage()(images[0])
plt.imshow(image)
plt.show()

# Calculates output layer size, convenient when making multiple hidden layers
def output_size(in_channels, kernel_size, padding, stride, pool):
    return np.floor((((in_channels - kernel_size + (2 * padding)) / stride) + 1) / pool)

# Test Run of output_size
output_shape = output_size(128, 3, 0, 1, 2)
print(output_shape)
