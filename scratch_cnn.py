
#!/usr/bin/env python
# coding: utf-8

# In[8]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms as T # for simplifying the transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models


# In[9]:


import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import sys
from tqdm import tqdm
import time
import copy


# In[22]:


def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
    return all_data.classes
def get_data_loaders(data_dir, batch_size, train = False):
    if train:
        #train
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.25),
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # imagenet means
            T.RandomErasing(p=0.2, value='random')
        ])
        train_data = datasets.ImageFolder(os.path.join(data_dir, "train/"), transform = transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        return train_loader, len(train_data)
    else:
        # val/test
        transform = T.Compose([ # We dont need augmentation for test transforms
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # imagenet means
        ])
        val_data = datasets.ImageFolder(os.path.join(data_dir, "val/"), transform=transform)
        test_data = datasets.ImageFolder(os.path.join(data_dir, "test/"), transform=transform)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
        return val_loader, test_loader, len(val_data), len(test_data)


# In[11]:


#get_ipython().system(' pip install split-folders')

import splitfolders


# In[19]:


input_folder = r'C:\\Users\\PC\\Desktop\\CS464\\real_and_fake_face'


# In[21]:


# splitfolders.ratio(input_folder, output=r"C:\\Users\\PC\\Desktop\\CS464\\real_and_fake_face_split", 
#                    seed=42, ratio=(.8, .1,.1), 
#                    group_prefix=None) # default values


# In[23]:


dataset_path = r"C:\\Users\\PC\\Desktop\\CS464\\real_and_fake_face_split"


# In[24]:


(train_loader, train_data_len) = get_data_loaders(dataset_path, 32, train=True)
(val_loader, test_loader, valid_data_len, test_data_len) = get_data_loaders(dataset_path, 32, train=False)


# In[27]:


classes = get_classes(r"C:\\Users\\PC\\Desktop\\CS464\\real_and_fake_face_split\\train/")
print(classes, len(classes))


# In[28]:


dataloaders = {
    "train": train_loader,
    "val": val_loader
}
dataset_sizes = {
    "train": train_data_len,
    "val": valid_data_len
}


# In[29]:
# if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda')
device


# In[30]:


import torch.nn as nn

"""
    Propsoed CNN architecture.
    
"""

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Pamameters Initialization
        input_shape = (3,224,224)
        activation = nn.ReLU()
        padding = 1
        droprate = 0.1
        epsilon=0.001

        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=input_shape[0]),
            nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=padding),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=16, eps=epsilon)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=padding),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32, eps=epsilon),
            nn.Dropout2d(p=droprate)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=padding),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=64, eps=epsilon),
            nn.Dropout2d(p=droprate)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=padding),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=128, eps=epsilon),
            nn.Dropout2d(p=droprate)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=padding),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=256, eps=epsilon),
            nn.Dropout2d(p=droprate)
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=padding),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=512, eps=epsilon),
            nn.Dropout2d(p=droprate)
        )

        self.layer7 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x

model = CNN()
model.to('cuda')
print(model) # Summary of the architecture


# In[32]:


from torch.optim import Adam
 
# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


# In[39]:


model


# In[40]:


model.train()


# In[49]:


count=1
for batch_idx, (data, target) in enumerate(train_loader):
    data = data.to('cuda')
    target = target.to('cuda')
    if count==1:
        i1,z1=batch_idx, (data, target)
        count+=1
    elif count==2:
        i2,z2=batch_idx, (data, target)
        count+=1
    elif count==3:
        i3,z3=batch_idx, (data, target)
        count+=1   
    elif count==4:
        i4,z4=batch_idx, (data, target)
        break       
    


# In[52]:

optimizer.zero_grad()


# In[53]:


output = model(z1[0])


# In[76]:


z1[1]


# In[ ]:





# In[82]:





# In[77]:


def loss_criteria(y_hat, y):
    return nn.BCELoss()(y_hat, y)


# In[83]:


loss = loss_criteria(output.resize(32).float(), target.float())


# In[87]:


len(output)


# In[88]:


def train(model, device, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # Use the CPU or GPU as appropriate
        # Recall that GPU is optimized for the operations we are dealing with
        data, target = data.to(device), target.to(device)
        
        # Reset the optimizer
        optimizer.zero_grad()
        
        # Push the data forward through the model layers
        output = model(data)
        output=output.to(torch.float32)
        target=target.to(torch.float32)
        
        # Get the loss
        loss = loss_criteria(output.resize(len(output)), target)

        # Keep a running total
        train_loss += loss.item()
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        
        # Print metrics so we see some progress
        print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))
            
    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss


# In[93]:


def test(model, device, test_loader):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            
            # Get the predicted classes for this batch
            output = model(data)
            output=output.to(torch.float32)
            target=target.to(torch.float32)
            
            # Calculate the loss for this batch
            test_loss += loss_criteria(output.resize(len(output)), target).item()
            
            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target==predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # return average loss for the epoch
    return avg_loss


# In[94]:


# Use an "Adam" optimizer to adjust weights
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Specify the loss criteria

# Track metrics in these arrays
epoch_nums = []
training_loss = []
validation_loss = []

# Train over 10 epochs (We restrict to 10 for time issues)
epochs = 1
print('Training on', device)
for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)


# In[97]:


import seaborn as sns

# Required magic to display matplotlib plots in notebooks
#get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

truelabels = []
predictions = []
model.eval()
from sklearn.metrics import accuracy_score, confusion_matrix
print("Getting predictions from test set...")
for data, target in test_loader:
    data = data.to('cuda')
    target = target.to('cuda')
    for label in target.cpu().data.numpy():
        truelabels.append(label)
    for prediction in model(data).cpu().data.numpy().argmax(1):
        predictions.append(prediction) 

# Plot the confusion matrix
cm = confusion_matrix(truelabels, predictions)
tick_marks = np.arange(len(classes))

df_cm = pd.DataFrame(cm, index = classes, columns = classes)
plt.figure(figsize = (7,7))
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
plt.xlabel("Predicted Shape", fontsize = 20)
plt.ylabel("True Shape", fontsize = 20)
plt.show()


# In[ ]:


