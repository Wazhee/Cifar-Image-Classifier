import torch
import numpy as np


class CIFARDataset(Dataset):
  def __init__(self, images, labels, mode, transform):
    self.transform = transform
    self.mode = mode

    if mode == 'train':
      self.images = images[:40000]
      self.labels = labels[:40000]
      # Your code here. If training, use the first 40,000 examples of the 
      # entire dataset 

    elif mode == 'val':
      self.images = images[40000:50000]
      self.labels = labels[40000:50000]
      # Your code here. If validation, use examples 40,000-50,000
      # of the entire dataset 
       
    elif mode == 'test':
      self.images = images[50000:60000]
      self.labels = labels[50000:60000]
      # Your code here. If testing, use examples 50,000-60,000 of the 
      # entire dataset 
      
    else:
      raise ValueError('Invalid mode!')

  def __getitem__(self, idx):
    # Do the following:
    # 1. Get the image and label from the dataset corresponding to index idx.]
    x, y = self.images[idx], self.labels[idx, ...]
    # 2. Convert the label to a LongTensor (needs to be of this type because it 
    # is an integer value and PyTorch will throw an error otherwise)
    label = torch.LongTensor(y)
    # 3. Transform the image using self.transform. This will convert the image 
    # into a tensor, scale it to [0,1], and apply data augmentations.
    image = self.transform(x)
    # # 4. Return the image and label.   
    return image, label

  def __len__(self):
    return len(self.labels) # Replace with your code.
